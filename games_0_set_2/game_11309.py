import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    GameEnv: Escape intricate, gravity-defying cube prisons by manipulating
    the environment and avoiding light beams as a shadowy entity.

    Visual Style: Minimalist, geometric, dark with contrasting neon light beams.
    Isometric perspective of a rotating cube.

    Core Gameplay:
    - The player (a glowing square) moves on the surface of a 3D cube.
    - Light beams (glowing red lines) sweep across the cube's faces.
    - The player can rotate the cube's internal geometry, which changes the
      player's position and the paths of the light beams.
    - The goal is to reach the exit portal (glowing white square).
    - Touching a light beam results in failure.

    Action Space (MultiDiscrete([5, 2, 2])):
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) relative
                  to the current face's orientation.
    - actions[1]: Space button (0=released, 1=held). A press rotates the
                  cube's contents 90 degrees around the Y-axis.
    - actions[2]: Shift button (0=released, 1=held). A press rotates the
                  cube's contents 90 degrees around the X-axis.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a shadowy entity across the faces of a rotating cube prison. "
        "Manipulate the cube's orientation to find the exit portal while avoiding deadly light beams."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. "
        "Press 'space' to rotate the cube horizontally and 'shift' to rotate it vertically."
    )
    auto_advance = True

    # --- Colors and Constants ---
    COLOR_BG = (26, 26, 46)
    COLOR_CUBE_FACE = (50, 50, 70, 150)
    COLOR_CUBE_EDGE = (150, 150, 180)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_EXIT = (255, 255, 255)
    COLOR_BEAM = (233, 69, 96)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SHADOW = (10, 10, 20)

    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    CUBE_HALF_SIZE = 1.0
    PLAYER_SPEED = 0.2
    PLAYER_SIZE = 0.15
    BEAM_WIDTH = 0.05
    MAX_STEPS = 2000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Consolas', 24, bold=True)
        
        # --- 3D Setup ---
        self.world_rotation = self._get_rotation_matrix(x_angle=math.pi/6, y_angle=-math.pi/4)
        self.cube_vertices = self._get_cube_vertices()
        self.cube_faces = [
            [0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], 
            [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]
        ]
        self.face_normals = np.array([
            [0, 0, -1], [0, 0, 1], [-1, 0, 0], 
            [1, 0, 0], [0, -1, 0], [0, 1, 0]
        ])
        # Face orientation vectors (up, right) in 3D local space
        self.face_orientations = {
            (0, 0, 1): ([0, 1, 0], [-1, 0, 0]),  # Front face (+Z)
            (0, 0, -1): ([0, 1, 0], [1, 0, 0]),   # Back face (-Z)
            (1, 0, 0): ([0, 1, 0], [0, 0, 1]),    # Right face (+X)
            (-1, 0, 0): ([0, 1, 0], [0, 0, -1]),  # Left face (-X)
            (0, 1, 0): ([0, 0, -1], [1, 0, 0]),   # Top face (+Y)
            (0, -1, 0): ([0, 0, 1], [1, 0, 0]),   # Bottom face (-Y)
        }

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_levels_cleared = 0
        self.level = 1
        
        self.player_pos_3d = np.zeros(3)
        self.exit_pos_3d = np.zeros(3)
        self.beams = []
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.prev_dist_to_exit = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        if options and "level_cleared" in options and options["level_cleared"]:
            self.total_levels_cleared += 1
            self.level = self.total_levels_cleared + 1
        else:
            self.total_levels_cleared = 0
            self.level = 1

        self._generate_level()
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.prev_dist_to_exit = self._get_distance_to_exit()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.1  # Cost of living

        # --- Unpack and Process Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        self._update_player_movement(movement)
        
        # Rotation on button press (rising edge)
        if space_held and not self.prev_space_held:
            self._rotate_cube_contents('y', -math.pi/2) # Clockwise
        if shift_held and not self.prev_shift_held:
            self._rotate_cube_contents('x', -math.pi/2) # Clockwise
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game Logic ---
        self._update_beams()
        
        # --- Calculate Rewards ---
        dist_to_exit = self._get_distance_to_exit()
        reward += (self.prev_dist_to_exit - dist_to_exit) * 0.1
        self.prev_dist_to_exit = dist_to_exit
        
        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self._check_player_on_exit():
            reward += 100
            self.score += 100
            terminated = True
        elif self._check_beam_collision():
            reward -= 10
            self.score -= 10
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True

        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # region # --- State Update and Logic ---
    def _generate_level(self):
        # Place exit
        exit_face_idx = self.np_random.integers(0, 6)
        exit_normal = self.face_normals[exit_face_idx]
        u, v = self.np_random.uniform(-0.8, 0.8, size=2)
        self.exit_pos_3d = self._get_3d_pos_from_face_uv(tuple(exit_normal), u, v)

        # Place player on a different face
        player_face_idx = exit_face_idx
        while player_face_idx == exit_face_idx:
            player_face_idx = self.np_random.integers(0, 6)
        player_normal = self.face_normals[player_face_idx]
        u, v = self.np_random.uniform(-0.8, 0.8, size=2)
        self.player_pos_3d = self._get_3d_pos_from_face_uv(tuple(player_normal), u, v)

        # Generate beams
        self.beams = []
        num_beams = 1 + (self.total_levels_cleared // 3)
        beam_speed = 0.01 + (self.total_levels_cleared // 5) * 0.05
        
        possible_faces = list(range(6))
        possible_faces.remove(player_face_idx)
        if exit_face_idx in possible_faces:
            possible_faces.remove(exit_face_idx)
        
        for _ in range(num_beams):
            if not possible_faces: break
            face_idx = self.np_random.choice(possible_faces)
            possible_faces.remove(face_idx)
            
            self.beams.append({
                "normal": self.face_normals[face_idx],
                "pivot": self._get_3d_pos_from_face_uv(tuple(self.face_normals[face_idx]), 0, 0),
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "speed": self.np_random.choice([-1, 1]) * beam_speed * self.np_random.uniform(0.8, 1.2),
                "length": self.CUBE_HALF_SIZE * 1.2
            })
            
    def _update_player_movement(self, movement):
        if movement == 0: return

        face_normal, _ = self._get_face_from_3d(self.player_pos_3d)
        up_vec, right_vec = self.face_orientations[tuple(face_normal)]
        
        delta = np.zeros(3)
        if movement == 1: delta = np.array(up_vec)
        elif movement == 2: delta = -np.array(up_vec)
        elif movement == 3: delta = -np.array(right_vec)
        elif movement == 4: delta = np.array(right_vec)

        self.player_pos_3d += delta * self.PLAYER_SPEED
        self.player_pos_3d = self._clamp_to_cube_surface(self.player_pos_3d)

    def _rotate_cube_contents(self, axis, angle):
        rot_mat = self._get_rotation_matrix(**{f'{axis}_angle': angle})
        
        self.player_pos_3d = self.player_pos_3d @ rot_mat.T
        self.exit_pos_3d = self.exit_pos_3d @ rot_mat.T
        for beam in self.beams:
            beam["normal"] = beam["normal"] @ rot_mat.T
            beam["pivot"] = beam["pivot"] @ rot_mat.T

    def _update_beams(self):
        for beam in self.beams:
            beam["angle"] += beam["speed"]

    def _check_player_on_exit(self):
        dist = np.linalg.norm(self.player_pos_3d - self.exit_pos_3d)
        return dist < self.PLAYER_SIZE * 2

    def _check_beam_collision(self):
        player_face_normal, (player_u, player_v) = self._get_face_from_3d(self.player_pos_3d)
        
        for beam in self.beams:
            if not np.allclose(player_face_normal, beam["normal"]):
                continue

            _, (pivot_u, pivot_v) = self._get_face_from_3d(beam["pivot"])
            
            relative_u = player_u - pivot_u
            relative_v = player_v - pivot_v
            
            beam_angle = beam["angle"]
            rotated_u = relative_u * math.cos(-beam_angle) - relative_v * math.sin(-beam_angle)
            rotated_v = relative_u * math.sin(-beam_angle) + relative_v * math.cos(-beam_angle)
            
            if (abs(rotated_u) < beam["length"] and abs(rotated_v) < (self.BEAM_WIDTH + self.PLAYER_SIZE) / 2):
                return True
        return False
    # endregion

    # region # --- 3D Math and Coordinate Helpers ---
    def _get_rotation_matrix(self, x_angle=0, y_angle=0, z_angle=0):
        cx, sx = math.cos(x_angle), math.sin(x_angle)
        cy, sy = math.cos(y_angle), math.sin(y_angle)
        cz, sz = math.cos(z_angle), math.sin(z_angle)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def _get_cube_vertices(self):
        s = self.CUBE_HALF_SIZE
        return np.array([
            [-s, -s, -s], [s, -s, -s], [-s, s, -s], [s, s, -s],
            [-s, -s, s], [s, -s, s], [-s, s, s], [s, s, s]
        ])

    def _clamp_to_cube_surface(self, pos):
        abs_pos = np.abs(pos)
        max_dim = np.argmax(abs_pos)
        if abs_pos[max_dim] > self.CUBE_HALF_SIZE:
            pos *= self.CUBE_HALF_SIZE / abs_pos[max_dim]
        return pos

    def _get_face_from_3d(self, pos_3d):
        axis = np.argmax(np.abs(pos_3d))
        sign = np.sign(pos_3d[axis])
        normal = np.zeros(3)
        normal[axis] = sign
        
        up_vec, right_vec = self.face_orientations[tuple(normal)]
        
        u = np.dot(pos_3d, right_vec)
        v = np.dot(pos_3d, up_vec)
        
        return normal, (u, v)

    def _get_3d_pos_from_face_uv(self, normal, u, v):
        up_vec, right_vec = self.face_orientations[normal]
        pos = np.array(normal) * self.CUBE_HALF_SIZE
        pos += np.array(right_vec) * u
        pos += np.array(up_vec) * v
        return pos
        
    def _project(self, points_3d):
        rotated_points = points_3d @ self.world_rotation.T
        projected = rotated_points[:, :2] * np.array([1, -1])
        scale = self.SCREEN_HEIGHT * 0.4
        offset = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        return projected * scale + offset

    def _get_distance_to_exit(self):
        return np.linalg.norm(self.player_pos_3d - self.exit_pos_3d)
    # endregion

    # region # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_particles()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        projected_vertices = self._project(self.cube_vertices)

        rotated_normals = self.face_normals @ self.world_rotation.T
        face_avg_z = [rotated_normals[i][2] for i in range(len(self.cube_faces))]
        sorted_face_indices = np.argsort(face_avg_z)
        
        for i in sorted_face_indices:
            face_vertex_indices = self.cube_faces[i]
            points = projected_vertices[face_vertex_indices]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CUBE_FACE)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CUBE_EDGE)

        entities_to_draw = []
        player_face_normal, _ = self._get_face_from_3d(self.player_pos_3d)
        player_depth = (self.player_pos_3d @ self.world_rotation.T)[2]
        entities_to_draw.append((player_depth, "player", self.player_pos_3d, player_face_normal))
        exit_face_normal, _ = self._get_face_from_3d(self.exit_pos_3d)
        exit_depth = (self.exit_pos_3d @ self.world_rotation.T)[2]
        entities_to_draw.append((exit_depth, "exit", self.exit_pos_3d, exit_face_normal))
        for beam in self.beams:
            beam_depth = (beam["pivot"] @ self.world_rotation.T)[2]
            entities_to_draw.append((beam_depth, "beam", beam, beam["normal"]))
        
        entities_to_draw.sort(key=lambda x: x[0], reverse=True)

        for depth, type, data, normal in entities_to_draw:
            rotated_normal = normal @ self.world_rotation.T
            if rotated_normal[2] < 0: continue

            if type == "player": self._render_shape_on_cube(data, self.PLAYER_SIZE, self.COLOR_PLAYER, True)
            elif type == "exit": self._render_shape_on_cube(data, self.PLAYER_SIZE * 1.2, self.COLOR_EXIT, True)
            elif type == "beam": self._render_beam(data)

    def _render_shape_on_cube(self, pos_3d, size, color, glow):
        _, (u, v) = self._get_face_from_3d(pos_3d)
        face_normal, _ = self._get_face_from_3d(pos_3d)
        up_vec, right_vec = self.face_orientations[tuple(face_normal)]
        
        corners_3d = [
            pos_3d + (up * v_off + right * u_off) * size for u_off, v_off in [(-1,-1), (1,-1), (1,1), (-1,1)]
            for up, right in [(np.array(up_vec), np.array(right_vec))]
        ]
        
        projected_corners = self._project(np.array(corners_3d))
        
        if glow: self._draw_glow_polygon(projected_corners, color)
        pygame.gfxdraw.filled_polygon(self.screen, projected_corners, color)
        pygame.gfxdraw.aapolygon(self.screen, projected_corners, color)

    def _render_beam(self, beam):
        face_normal, (pivot_u, pivot_v) = self._get_face_from_3d(beam["pivot"])
        up_vec, right_vec = self.face_orientations[tuple(face_normal)]
        
        angle, length = beam["angle"], beam["length"]
        u1 = pivot_u + length * math.cos(angle)
        v1 = pivot_v + length * math.sin(angle)
        u2 = pivot_u - length * math.cos(angle)
        v2 = pivot_v - length * math.sin(angle)
        
        p1_3d = self._get_3d_pos_from_face_uv(tuple(face_normal), u1, v1)
        p2_3d = self._get_3d_pos_from_face_uv(tuple(face_normal), u2, v2)
        
        p1_2d, p2_2d = self._project(np.array([p1_3d, p2_3d]))
        
        self._draw_glow_line(p1_2d, p2_2d, self.COLOR_BEAM, int(self.BEAM_WIDTH * 30))
        
        pygame.draw.line(self.screen, self.COLOR_BEAM, p1_2d, p2_2d, int(self.BEAM_WIDTH * 20))

    def _render_ui(self):
        level_text = f"Level: {self.level}"
        score_text = f"Score: {self.score:.1f}"
        
        self._draw_text(level_text, (10, 10))
        self._draw_text(score_text, (self.SCREEN_WIDTH - 10, 10), align="right")

    def _draw_text(self, text, pos, align="left"):
        text_surf = self.font.render(text, True, self.COLOR_TEXT)
        shadow_surf = self.font.render(text, True, self.COLOR_SHADOW)
        
        rect = text_surf.get_rect()
        if align == "right": rect.topright = pos
        else: rect.topleft = pos
        
        shadow_rect = rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, rect)

    def _draw_glow_polygon(self, points, color):
        surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for i in range(4, 0, -1):
            alpha = 80 - i * 15
            poly_color = (*color, alpha)
            
            center = np.mean(points, axis=0)
            glow_points = [center + (p - center) * (1 + i * 0.15) for p in points]
            
            pygame.gfxdraw.filled_polygon(surf, glow_points, poly_color)
            pygame.gfxdraw.aapolygon(surf, glow_points, poly_color)
        self.screen.blit(surf, (0, 0))

    def _draw_glow_line(self, p1, p2, color, width):
        for i in range(5, 0, -1):
            alpha = 100 - i * 15
            glow_color = (*color, alpha)
            pygame.draw.line(self.screen, glow_color, p1, p2, width + i * 4)

    def _render_background_particles(self):
        if not hasattr(self, 'particles'):
            self.particles = []
            for _ in range(50):
                self.particles.append({
                    'pos': [random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)],
                    'vel': [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)],
                    'radius': random.uniform(0.5, 1.5),
                    'color': (70, 70, 90)
                })
        
        for p in self.particles:
            p['pos'][0] = (p['pos'][0] + p['vel'][0]) % self.SCREEN_WIDTH
            p['pos'][1] = (p['pos'][1] + p['vel'][1]) % self.SCREEN_HEIGHT
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

    # endregion

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "distance_to_exit": self.prev_dist_to_exit
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Requires pygame to be installed with display support.
    # To run, unset the dummy video driver:
    # del os.environ['SDL_VIDEODRIVER']
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Pygame setup for manual play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cube Prison")
    clock = pygame.time.Clock()

    while running:
        if terminated:
            # On win/loss, reset the environment
            print(f"Episode finished. Final Score: {info['score']:.2f}")
            was_win = env._check_player_on_exit()
            obs, info = env.reset(options={"level_cleared": was_win})
            terminated = False

        # --- Action mapping for human play ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    env.close()