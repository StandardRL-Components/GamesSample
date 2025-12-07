import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:11:05.945660
# Source Brief: brief_00862.md
# Brief Index: 862
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    Chroma Maze - A Gymnasium environment where the agent guides a transformable light beam
    through a maze of obstacles and mirrors to reach a target destination.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) - Moves the beam emitter.
    - actions[1]: Wavelength Switch (0=released, 1=pressed) - Toggles between Red and Blue.
    - actions[2]: Fire Beam (0=released, 1=pressed) - Fires the beam for one reflection sequence.

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - An RGB image of the game screen.

    Reward Structure:
    - +100: Reaching the target.
    - +10: Opening a path with a blue beam.
    - +5: Destroying an obstacle with a red beam.
    - +0.1: For firing the beam (successful reflection).
    - -10: Hitting an obstacle with the wrong color beam.
    - -50: 10% chance on a wrong-color hit, representing a major mistake.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Guide a transformable light beam through a maze of obstacles and mirrors to reach the target."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move the emitter. Press space to switch beam color and shift to fire the beam."
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_REFLECTIONS = 40
    MAX_STEPS = 1000
    PLAYER_SPEED = 8
    BEAM_WIDTH = 3

    # --- Colors ---
    COLOR_BG = (15, 18, 32)
    COLOR_WALL = (40, 45, 60)
    COLOR_MIRROR = (180, 190, 210)
    COLOR_TARGET = (255, 215, 0)
    COLOR_PLAYER = (230, 230, 255)

    COLOR_BEAM_RED = (255, 50, 50)
    COLOR_BEAM_BLUE = (50, 150, 255)

    OBSTACLE_RED = 0
    OBSTACLE_BLUE = 1
    COLOR_OBSTACLE_RED = (200, 60, 60)
    COLOR_OBSTACLE_BLUE = (60, 120, 200)

    WAVELENGTH_RED = 0
    WAVELENGTH_BLUE = 1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_wavelength = pygame.font.SysFont("Consolas", 16, bold=True)

        # --- Game State Initialization ---
        self.player_pos = None
        self.wavelength = None
        self.reflection_count = None
        self.steps = None
        self.score = None
        self.game_over = None

        self.walls = []
        self.mirrors = []
        self.obstacles = []
        self.target = None

        self.beam_segments = []
        self.particles = []

        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Core State ---
        self.steps = 0
        self.score = 0
        self.reflection_count = 0
        self.game_over = False
        self.wavelength = self.WAVELENGTH_RED

        # --- Reset Button States ---
        self.prev_space_held = False
        self.prev_shift_held = False

        # --- Clear Dynamic Elements ---
        self.beam_segments.clear()
        self.particles.clear()

        # --- Level Layout (Fixed for consistency) ---
        self.player_pos = np.array([50.0, self.HEIGHT / 2.0])

        self.walls = [
            pygame.Rect(0, 0, self.WIDTH, 10),
            pygame.Rect(0, self.HEIGHT - 10, self.WIDTH, 10),
            pygame.Rect(0, 0, 10, self.HEIGHT),
            pygame.Rect(self.WIDTH - 10, 0, 10, self.HEIGHT),
            pygame.Rect(150, 100, 20, 200)
        ]

        self.mirrors = [
            {'rect': pygame.Rect(300, 50, 10, 100), 'angle': math.radians(45)},
            {'rect': pygame.Rect(450, 250, 100, 10), 'angle': math.radians(-25)}
        ]

        self.obstacles = [
            {'rect': pygame.Rect(220, 80, 20, 20), 'type': self.OBSTACLE_RED, 'active': True},
            {'rect': pygame.Rect(220, 300, 20, 20), 'type': self.OBSTACLE_BLUE, 'active': True},
            {'rect': pygame.Rect(400, 150, 20, 20), 'type': self.OBSTACLE_RED, 'active': True},
        ]

        self.target = pygame.Rect(self.WIDTH - 60, self.HEIGHT / 2 - 15, 30, 30)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = False
        self.steps += 1

        # --- Unpack and Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # 1. Player (Emitter) Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)
        self.player_pos[1] = np.clip(self.player_pos[1], 20, self.HEIGHT - 20)

        # 2. Wavelength Switch (on button press)
        if space_held and not self.prev_space_held:
            self.wavelength = self.WAVELENGTH_BLUE if self.wavelength == self.WAVELENGTH_RED else self.WAVELENGTH_RED

        # 3. Fire Beam (on button press)
        if shift_held and not self.prev_shift_held:
            if self.reflection_count < self.MAX_REFLECTIONS:
                self.reflection_count += 1
                fire_reward, hit_target = self._fire_beam()
                reward += fire_reward
                if hit_target:
                    self.game_over = True
            else:
                self.game_over = True # Out of reflections
        else:
            # Clear beam if fire button is not held.
            self.beam_segments.clear()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        self._update_particles()

        self.score += reward

        terminated = self.game_over or self.steps >= self.MAX_STEPS or self.reflection_count >= self.MAX_REFLECTIONS
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _fire_beam(self):
        self.beam_segments.clear()

        beam_origin = self.player_pos.copy()
        # Fire right by default
        beam_dir = np.array([1.0, 0.0])

        total_reward = 0.1 # Small reward for firing
        hit_target = False

        MAX_BOUNCES = 10 # Prevent infinite loops
        for _ in range(MAX_BOUNCES):
            hit = self._cast_ray(beam_origin, beam_dir)

            if hit is None:
                end_point = beam_origin + beam_dir * 2000
                self.beam_segments.append((beam_origin, end_point))
                break

            hit_point, hit_obj_type, hit_obj_data = hit
            self.beam_segments.append((beam_origin, hit_point))

            if hit_obj_type == 'wall':
                self._create_particles(hit_point, 10, (100, 100, 100))
                break

            elif hit_obj_type == 'target':
                total_reward += 100
                hit_target = True
                self._create_particles(hit_point, 50, self.COLOR_TARGET, 5, 2)
                break

            elif hit_obj_type == 'obstacle':
                obstacle = hit_obj_data
                if obstacle['active']:
                    correct_wavelength = (self.wavelength == self.WAVELENGTH_RED and obstacle['type'] == self.OBSTACLE_RED) or \
                                         (self.wavelength == self.WAVELENGTH_BLUE and obstacle['type'] == self.OBSTACLE_BLUE)

                    if correct_wavelength:
                        obstacle['active'] = False
                        if obstacle['type'] == self.OBSTACLE_RED:
                            total_reward += 5
                            self._create_particles(hit_point, 30, self.COLOR_OBSTACLE_RED)
                        else: # Blue
                            total_reward += 10
                            self._create_particles(hit_point, 30, self.COLOR_OBSTACLE_BLUE)
                    else:
                        total_reward -= 10
                        if self.np_random.random() < 0.1:
                            total_reward -= 50
                        self._create_particles(hit_point, 20, (50, 50, 50), 2, 0.5)
                break

            elif hit_obj_type == 'mirror':
                self._create_particles(hit_point, 5, self.COLOR_MIRROR, 1)
                normal = np.array([math.cos(hit_obj_data['angle'] + math.pi/2), math.sin(hit_obj_data['angle'] + math.pi/2)])
                beam_dir = beam_dir - 2 * np.dot(beam_dir, normal) * normal
                beam_origin = hit_point + beam_dir * 1e-3
            else:
                break

        return total_reward, hit_target

    def _cast_ray(self, origin, direction):
        closest_hit = None
        min_dist = float('inf')

        collidables = [('wall', w) for w in self.walls] + \
                      [('mirror', m) for m in self.mirrors] + \
                      [('obstacle', o) for o in self.obstacles if o['active']] + \
                      [('target', self.target)]

        for obj_type, obj_data in collidables:
            rect = obj_data if obj_type in ['wall', 'target'] else obj_data['rect']
            
            t_near = -float('inf')
            t_far = float('inf')

            for i in range(2):
                if abs(direction[i]) < 1e-6:
                    if not (rect.topleft[i] <= origin[i] <= rect.bottomright[i]):
                        continue
                
                t1 = (rect.topleft[i] - origin[i]) / direction[i]
                t2 = (rect.bottomright[i] - origin[i]) / direction[i]

                if t1 > t2: t1, t2 = t2, t1
                t_near = max(t_near, t1)
                t_far = min(t_far, t2)

                if t_near > t_far or t_far < 0:
                    break
            else:
                if t_near >= 0 and t_near < min_dist:
                    min_dist = t_near
                    hit_point = origin + direction * t_near
                    closest_hit = (hit_point, obj_type, obj_data)

        return closest_hit

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "reflections": self.reflection_count}

    def _render_game(self):
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        for mirror in self.mirrors:
            angle = mirror['angle']
            center = np.array(mirror['rect'].center)
            half_len = mirror['rect'].width / 2 if mirror['rect'].width > mirror['rect'].height else mirror['rect'].height / 2
            start = center + half_len * np.array([math.cos(angle), math.sin(angle)])
            end = center - half_len * np.array([math.cos(angle), math.sin(angle)])
            pygame.draw.line(self.screen, self.COLOR_MIRROR, start, end, 4)

        for obs in self.obstacles:
            if obs['active']:
                color = self.COLOR_OBSTACLE_RED if obs['type'] == self.OBSTACLE_RED else self.COLOR_OBSTACLE_BLUE
                glow_color = tuple(min(255, c + 50) for c in color)
                pygame.gfxdraw.box(self.screen, obs['rect'].inflate(6, 6), (*glow_color, 60))
                pygame.draw.rect(self.screen, color, obs['rect'])

        pygame.gfxdraw.box(self.screen, self.target.inflate(8, 8), (*self.COLOR_TARGET, 80))
        pygame.draw.rect(self.screen, self.COLOR_TARGET, self.target)
        pygame.draw.rect(self.screen, self.COLOR_BG, self.target.inflate(-10, -10))

        self._render_particles()
        self._render_beam()

        player_x, player_y = int(self.player_pos[0]), int(self.player_pos[1])
        wavelength_color = self.COLOR_BEAM_RED if self.wavelength == self.WAVELENGTH_RED else self.COLOR_BEAM_BLUE
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        glow_radius = int(12 + pulse * 4)
        glow_alpha = int(50 + pulse * 20)
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, glow_radius, (*wavelength_color, glow_alpha))
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, 8, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, 8, self.COLOR_PLAYER)

    def _render_beam(self):
        if not self.beam_segments:
            return
        color = self.COLOR_BEAM_RED if self.wavelength == self.WAVELENGTH_RED else self.COLOR_BEAM_BLUE
        glow_color = (*color, 70)
        for start, end in self.beam_segments:
            pygame.draw.line(self.screen, color, start, end, self.BEAM_WIDTH)
            pygame.draw.line(self.screen, glow_color, start, end, self.BEAM_WIDTH * 3)

    def _render_particles(self):
        for p in self.particles:
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                rect = pygame.Rect(int(p['pos'][0] - size/2), int(p['pos'][1] - size/2), size, size)
                pygame.draw.rect(self.screen, p['color'], rect)

    def _render_ui(self):
        refl_text = f"REFLECTIONS: {self.reflection_count}/{self.MAX_REFLECTIONS}"
        score_text = f"SCORE: {self.score:.1f}"
        refl_surf = self.font_ui.render(refl_text, True, (200, 200, 220))
        score_surf = self.font_ui.render(score_text, True, (200, 200, 220))
        self.screen.blit(refl_surf, (20, 20))
        self.screen.blit(score_surf, (20, 45))

        wavelength_str = "RED" if self.wavelength == self.WAVELENGTH_RED else "BLUE"
        color = self.COLOR_BEAM_RED if self.wavelength == self.WAVELENGTH_RED else self.COLOR_BEAM_BLUE
        box_rect = pygame.Rect(self.WIDTH - 130, self.HEIGHT - 40, 110, 30)
        pygame.draw.rect(self.screen, (30, 35, 50), box_rect)
        pygame.draw.rect(self.screen, color, box_rect, 2)
        text_surf = self.font_wavelength.render(f"WAVELENGTH: {wavelength_str}", True, (220, 220, 240))
        self.screen.blit(text_surf, (box_rect.x + 10, box_rect.y + 7))

    def _create_particles(self, pos, count, color, speed=3, lifespan=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(0.5, 1.0) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': lifespan,
                'max_life': lifespan,
                'color': color,
                'size': self.np_random.uniform(3, 6)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 1]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Create a display for manual playing
    # This check is to allow the script to run headlessly for testing
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        print("Running in headless mode. Manual play is disabled.")
    else:
        pygame.display.set_caption("Chroma Maze - Manual Control")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        clock = pygame.time.Clock()
        
        print("\n--- Manual Control ---")
        print("Arrows: Move emitter")
        print("Space: Switch wavelength")
        print("Shift: Fire beam")
        print("Q: Quit")
        
        while not terminated:
            # --- Action Mapping for Manual Control ---
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reflections: {info['reflections']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    terminated = True
                    
            # --- Rendering to Display ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(env.FPS)

        print("\nGame Over!")
        print(f"Final Score: {info['score']:.2f}")
        print(f"Total Reflections: {info['reflections']}")
    
    env.close()