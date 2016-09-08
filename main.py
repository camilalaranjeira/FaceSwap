##################################
# Author: Camila Laranjeira
# July 7th, 2016
# References: http://www.learnopencv.com/face-swap-using-opencv-c-python/
#             https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
##################################

import dlib
import cv2
import numpy
from skimage import io

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        print "TooManyFaces"
    if len(rects) == 0:
        print "NoFaces"

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True

#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList()

    delaunayTri = []
    pt = []
    count= 0
    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            count = count + 1
            ind = []
            for j in xrange(0, 3):
                for k in xrange(0, len(points)):
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))
        pt = []

    return delaunayTri

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(numpy.float32(srcTri), numpy.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(numpy.float32([t1]))
    r2 = cv2.boundingRect(numpy.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in xrange(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = numpy.zeros((r2[3], r2[2], 3), dtype = numpy.float32)
    cv2.fillConvexPoly(mask, numpy.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

def realTimeLandmarks():
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = get_landmarks(frame)
        landmarks = numpy.squeeze(numpy.asarray(landmarks))
        n = 0
        for l in landmarks:
            cv2.circle(frame, (l[0], l[1]), 1, (0, 0, 255), 2)
            cv2.putText(frame, str(n), (l[0], l[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
            n += 1

        cv2.imshow("face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
#    realTimeLandmarks()
#    exit()

    img1 = cv2.imread("esposa-6.png", cv2.IMREAD_COLOR)
    #cap = cv2.VideoCapture(0)

    #while(True):
    img2 = cv2.imread("cunha.jpg", cv2.IMREAD_COLOR)
 #   ret, img2 = cap.read()
    img1Warped = numpy.copy(img2)

    landmarks_img1 = get_landmarks(img1)
    landmarks_img2 = get_landmarks(img2)

    landmarks_img1 = numpy.squeeze(numpy.asarray(landmarks_img1))
    landmarks_img2 = numpy.squeeze(numpy.asarray(landmarks_img2))

    # --------------------------------------------------- #
    # Show landmarks
    img1_landmarks = numpy.copy(img1)
    img2_landmarks = numpy.copy(img2)
    n = 0
    for l in landmarks_img1:
        cv2.circle(img1_landmarks, (l[0], l[1]), 1, (0, 0, 255), 2)
        cv2.putText(img1_landmarks, str(n),(l[0], l[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255,255))
        n += 1

    n = 0
    for l in landmarks_img2:
        cv2.circle(img2_landmarks, (l[0], l[1]), 1, (0, 0, 255), 2)
        cv2.putText(img2_landmarks, str(n), (l[0], l[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
        n += 1
    # --------------------------------------------------- #

    # --------------------------------------------------- #
    # Find convex hull (boundaries) and delaunay triangles #
    # Using file points because convexhull selected wrong region
    points = numpy.loadtxt("points.txt", delimiter=',')
    points = points.astype(int)
    coords = [(numpy.squeeze(landmarks_img1[p])[0], numpy.squeeze(landmarks_img1[p])[1]) for p in points]

    # hullIndex = cv2.convexHull(landmarks_img1, returnPoints=False)
    # coords = [(numpy.squeeze(landmarks_img1[h])[0], numpy.squeeze(landmarks_img1[h])[1]) for h in hullIndex]
    sizeImg = img1.shape
    rect = (0, 0, sizeImg[1], sizeImg[0])
    delaunay = calculateDelaunayTriangles(rect, coords)

    print delaunay
    # --------------------------------------------------- #


    img1_delaunay = numpy.copy(img1_landmarks)
    img2_delaunay = numpy.copy(img2_landmarks)
    for d in delaunay:
        print d
        t1 = [(numpy.squeeze(landmarks_img1[coor])[0], numpy.squeeze(landmarks_img1[coor])[1])for coor in d]
        t2 = [(numpy.squeeze(landmarks_img2[coor])[0], numpy.squeeze(landmarks_img2[coor])[1]) for coor in d]

        # --------------------------------------------------- #
        # Show delaunay triangulation
        cv2.line(img1_delaunay, t1[0], t1[1], (255, 255, 255))
        cv2.line(img1_delaunay, t1[1], t1[2], (255, 255, 255))
        cv2.line(img1_delaunay, t1[2], t1[0], (255, 255, 255))

        cv2.line(img2_delaunay, t2[0], t2[1], (255, 255, 255))
        cv2.line(img2_delaunay, t2[1], t2[2], (255, 255, 255))
        cv2.line(img2_delaunay, t2[2], t2[0], (255, 255, 255))
        # --------------------------------------------------- #

        warpTriangle(img1, img1Warped, t1, t2)
    # Calculate Mask
    hull8U = [(numpy.squeeze(landmarks_img2[p])[0], numpy.squeeze(landmarks_img2[p])[1]) for p in points]
    hull32 = [(numpy.squeeze(landmarks_img2[p])[0], numpy.squeeze(landmarks_img2[p])[1]) for p in points]
    mask = numpy.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, numpy.int32(hull8U), (255, 255, 255))
    r = cv2.boundingRect(numpy.float32([hull32]))
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    # Clone seamlessly.
    output = cv2.seamlessClone(numpy.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
    cv2.imshow("landmarks", img1_landmarks)
    cv2.imshow("Delaunay", img1_delaunay)
    cv2.imshow("landmarks2", img2_landmarks)
    cv2.imshow("Delaunay2", img2_delaunay)
    cv2.imshow("Warp", img1Warped)
    cv2.imshow("Output", output)
    cv2.waitKey()
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

    cv2.destroyAllWindows()