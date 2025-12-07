import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the goal is to guide a laser beam through a maze of
    rotatable mirrors to hit a target. The score is based on the number of reflections.
    The game is played in real-time with a time limit.

    Visual Style:
    - Clean, geometric, neon-inspired visuals on a dark background.
    - The laser has a bright cyan glow.
    - The target pulses with a green light.
    - The currently selected mirror is highlighted.

    Gameplay:
    - The player/agent controls the mirrors.
    - Use Left/Right actions to select a mirror.
    - Use Up/Down actions to rotate the selected mirror.
    - The laser path is calculated and displayed in real-time.
    - A high score is achieved by creating a long reflection path before hitting the target.
    
    Action Space: MultiDiscrete([5, 2, 2])
    - action[0] (Movement): 0:None, 1:Up(RotCW), 2:Down(RotCCW), 3:Left(PrevMirror), 4:Right(NextMirror)
    - action[1] (Space): Held to provide a visual boost to the laser.
    - action[2] (Shift): No effect.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Guide a laser through a maze of rotatable mirrors to hit a target, maximizing reflections for a high score before time runs out."
    user_guide = "Use ←→ to select a mirror and ↑↓ to rotate it. Hold space for a visual laser boost."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 900  # 30 seconds at 30 FPS

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_LASER = (0, 255, 255)
    COLOR_LASER_GLOW = (0, 128, 128)
    COLOR_MIRROR = (180, 180, 200)
    COLOR_MIRROR_SELECTED = (255, 255, 255)
    COLOR_MIRROR_GLOW = (100, 100, 120)
    COLOR_TARGET = (0, 255, 128)
    COLOR_TARGET_GLOW = (0, 128, 64)
    COLOR_TEXT = (220, 220, 240)
    
    NUM_MIRRORS = 12
    MIRROR_WIDTH = 60
    ROTATION_SPEED = math.radians(2.0) # Radians per step
    MAX_REFLECTIONS = 50

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("Consolas", 20, bold=True)
        except pygame.error:
            self.font = pygame.font.Font(None, 24)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.laser_origin = None
        self.laser_initial_angle = None
        self.target_pos = None
        self.target_radius = None
        self.mirrors = []
        self.selected_mirror_idx = 0
        self.laser_path = []
        self.reflection_count = 0
        self.space_held_visual_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.laser_origin = pygame.math.Vector2(30, self.HEIGHT / 2)
        self.laser_initial_angle = math.radians(20)

        self.target_pos = pygame.math.Vector2(self.WIDTH - 40, self.HEIGHT / 2)
        self.target_radius = 12

        self.mirrors = self._generate_mirrors()
        self.selected_mirror_idx = 0
        
        self._calculate_laser_path()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        
        self.space_held_visual_timer = 5 if space_held else max(0, self.space_held_visual_timer - 1)

        # 1. Handle player action
        self._handle_action(movement)

        # 2. Update game state
        hit_target, hit_wall = self._calculate_laser_path()
        self.score = self.reflection_count * 10

        # 3. Calculate reward and termination
        terminated = self._check_termination(hit_target)
        reward = self._calculate_reward(terminated, hit_target)
        
        if terminated:
            self.game_over = True

        truncated = False
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_action(self, movement):
        if movement == 1:  # Up -> Rotate CW
            self.mirrors[self.selected_mirror_idx]['angle'] += self.ROTATION_SPEED
        elif movement == 2:  # Down -> Rotate CCW
            self.mirrors[self.selected_mirror_idx]['angle'] -= self.ROTATION_SPEED
        elif movement == 3:  # Left -> Previous Mirror
            self.selected_mirror_idx = (self.selected_mirror_idx - 1) % self.NUM_MIRRORS
        elif movement == 4:  # Right -> Next Mirror
            self.selected_mirror_idx = (self.selected_mirror_idx + 1) % self.NUM_MIRRORS
        
        # Normalize angle
        if movement in [1, 2]:
            self.mirrors[self.selected_mirror_idx]['angle'] %= (2 * math.pi)


    def _calculate_laser_path(self):
        self.laser_path = []
        self.reflection_count = 0
        
        ray_origin = pygame.math.Vector2(self.laser_origin)
        ray_dir = pygame.math.Vector2(1, 0).rotate_rad(self.laser_initial_angle)
        
        hit_target = False
        hit_wall = False

        for _ in range(self.MAX_REFLECTIONS):
            intersections = []

            # Check for mirror intersections
            for i, mirror in enumerate(self.mirrors):
                p1 = mirror['center'] + pygame.math.Vector2(self.MIRROR_WIDTH / 2, 0).rotate_rad(mirror['angle'])
                p2 = mirror['center'] - pygame.math.Vector2(self.MIRROR_WIDTH / 2, 0).rotate_rad(mirror['angle'])
                
                intersect_pt = self._ray_segment_intersection(ray_origin, ray_dir, p1, p2)
                if intersect_pt:
                    dist = ray_origin.distance_to(intersect_pt)
                    intersections.append({'dist': dist, 'pt': intersect_pt, 'type': 'mirror', 'mirror_idx': i})

            # Check for wall intersections
            wall_points = [
                (pygame.math.Vector2(0, 0), pygame.math.Vector2(self.WIDTH, 0)),
                (pygame.math.Vector2(self.WIDTH, 0), pygame.math.Vector2(self.WIDTH, self.HEIGHT)),
                (pygame.math.Vector2(self.WIDTH, self.HEIGHT), pygame.math.Vector2(0, self.HEIGHT)),
                (pygame.math.Vector2(0, self.HEIGHT), pygame.math.Vector2(0, 0)),
            ]
            for p1, p2 in wall_points:
                intersect_pt = self._ray_segment_intersection(ray_origin, ray_dir, p1, p2)
                if intersect_pt:
                    dist = ray_origin.distance_to(intersect_pt)
                    intersections.append({'dist': dist, 'pt': intersect_pt, 'type': 'wall'})
            
            # Check for target intersection
            intersect_pt = self._ray_circle_intersection(ray_origin, ray_dir, self.target_pos, self.target_radius)
            if intersect_pt:
                dist = ray_origin.distance_to(intersect_pt)
                intersections.append({'dist': dist, 'pt': intersect_pt, 'type': 'target'})

            if not intersections:
                hit_wall = True # Ray goes on forever without hitting anything
                break

            # Find closest intersection
            closest = min(intersections, key=lambda x: x['dist'])
            
            self.laser_path.append((ray_origin, closest['pt']))
            ray_origin = closest['pt']

            if closest['type'] == 'mirror':
                self.reflection_count += 1
                mirror_angle = self.mirrors[closest['mirror_idx']]['angle']
                normal = pygame.math.Vector2(0, 1).rotate_rad(mirror_angle)
                ray_dir = ray_dir.reflect(normal)
            elif closest['type'] == 'target':
                hit_target = True
                break
            elif closest['type'] == 'wall':
                hit_wall = True
                break
        
        return hit_target, hit_wall

    def _check_termination(self, hit_target):
        # Hitting a wall is not a terminal state, allowing the user to keep adjusting.
        # The game ends only when the target is hit or the time runs out.
        if hit_target or self.steps >= self.MAX_STEPS:
            return True
        return False

    def _calculate_reward(self, terminated, hit_target):
        if not terminated:
            return self.reflection_count * 0.1

        if hit_target:
            if self.score > 500:
                return 60.0
            else:
                return -100.0
        else: # Timeout
            return -100.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        self._render_target()
        self._render_mirrors()
        self._render_laser()

    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_target(self):
        pulse = abs(math.sin(self.steps * 0.1)) * 4
        
        # Glow
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.target_pos.x), int(self.target_pos.y),
            int(self.target_radius + pulse), self.COLOR_TARGET_GLOW
        )
        # Core
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.target_pos.x), int(self.target_pos.y),
            self.target_radius, self.COLOR_TARGET
        )
        pygame.gfxdraw.aacircle(
            self.screen, int(self.target_pos.x), int(self.target_pos.y),
            self.target_radius, self.COLOR_TARGET
        )


    def _render_mirrors(self):
        for i, mirror in enumerate(self.mirrors):
            p1 = mirror['center'] + pygame.math.Vector2(self.MIRROR_WIDTH / 2, 0).rotate_rad(mirror['angle'])
            p2 = mirror['center'] - pygame.math.Vector2(self.MIRROR_WIDTH / 2, 0).rotate_rad(mirror['angle'])
            
            color = self.COLOR_MIRROR_SELECTED if i == self.selected_mirror_idx else self.COLOR_MIRROR
            glow_color = self.COLOR_MIRROR_GLOW if i == self.selected_mirror_idx else (0,0,0,0)

            if i == self.selected_mirror_idx:
                pygame.draw.line(self.screen, glow_color, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 8)

            pygame.draw.line(self.screen, color, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 4)

    def _render_laser(self):
        if self.space_held_visual_timer > 0:
            width_mod = 2
            glow_mod = 4
        else:
            width_mod = 0
            glow_mod = 0

        for start, end in self.laser_path:
            # Glow
            pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, 
                             (int(start.x), int(start.y)), (int(end.x), int(end.y)), 6 + glow_mod)
            # Core
            pygame.draw.line(self.screen, self.COLOR_LASER, 
                             (int(start.x), int(start.y)), (int(end.x), int(end.y)), 2 + width_mod)


    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        time_text = self.font.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        reflect_text = self.font.render(f"REFLECTIONS: {self.reflection_count}", True, self.COLOR_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (10, 35))
        self.screen.blit(reflect_text, (10, 60))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "reflections": self.reflection_count}

    def _generate_mirrors(self):
        mirrors = []
        # Pre-defined layout for a consistent puzzle
        positions = [
            (120, 80), (250, 150), (150, 300),
            (400, 50), (500, 200), (450, 350),
            (300, 280), (200, 50), (550, 80),
            (80, 200), (350, 200), (520, 300)
        ]
        angles = [45, -30, 60, 10, -60, 120, 80, 135, -10, 90, 0, -45]
        
        for i in range(self.NUM_MIRRORS):
            pos = positions[i]
            angle = angles[i]
            mirrors.append({
                'center': pygame.math.Vector2(pos),
                'angle': math.radians(angle)
            })
        return mirrors

    def _ray_segment_intersection(self, ray_origin, ray_dir, p1, p2):
        v1 = ray_origin - p1
        v2 = p2 - p1
        v3 = pygame.math.Vector2(-ray_dir.y, ray_dir.x)
        
        dot_v2_v3 = v2.dot(v3)
        if abs(dot_v2_v3) < 1e-6: # Parallel lines
            return None

        t1 = v2.cross(v1) / dot_v2_v3
        t2 = v1.dot(v3) / dot_v2_v3

        if t1 >= 0.001 and 0 <= t2 <= 1: # t1 check to avoid self-intersection
            return ray_origin + t1 * ray_dir
        return None
        
    def _ray_circle_intersection(self, ray_origin, ray_dir, circle_center, radius):
        oc = ray_origin - circle_center
        a = ray_dir.dot(ray_dir)
        b = 2.0 * oc.dot(ray_dir)
        c = oc.dot(oc) - radius*radius
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None
        else:
            t1 = (-b - math.sqrt(discriminant)) / (2.0*a)
            t2 = (-b + math.sqrt(discriminant)) / (2.0*a)
            if t1 >= 0.001:
                return ray_origin + t1 * ray_dir
            if t2 >= 0.001:
                return ray_origin + t2 * ray_dir
            return None

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows for manual play and visualization.
    # To run, you may need to specify a non-dummy video driver e.g.:
    # SDL_VIDEODRIVER=x11 python your_file.py
    # or remove the os.environ call at the top of the file.
    
    # Forcing a visible driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "pygame" 
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Laser Reflection Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = [0, 0, 0] # No-op, no space, no shift
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Reward: {reward}")
            obs, info = env.reset()

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()