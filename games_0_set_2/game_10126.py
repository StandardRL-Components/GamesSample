import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:58:02.333160
# Source Brief: brief_00126.md
# Brief Index: 126
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player guides a fading light beam.
    The goal is to bounce the beam off mirrors to collect 10 crystals
    before the beam's energy, which depletes over time and with each bounce,
    runs out.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=unused, 2=unused, 3=rotate left, 4=rotate right)
    - actions[1]: Space button (0=released, 1=pressed to fire beam)
    - actions[2]: Shift button (0=released, 1=held, currently unused)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a fading light beam by rotating and firing it to bounce off mirrors. "
        "Collect all the crystals before your beam's energy runs out."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to aim the beam and press space to fire. "
        "Each bounce consumes energy."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 45 * FPS  # 45 seconds
    MAX_CRYSTALS = 10
    ROTATION_SPEED = math.radians(4)
    BOUNCE_ENERGY_COST = 0.07 # 7% energy cost per bounce

    # --- Colors ---
    COLOR_BG = (5, 5, 20)
    COLOR_MIRROR = (220, 220, 240)
    COLOR_MIRROR_GLOW = (150, 150, 180)
    COLOR_CRYSTAL = (0, 255, 255)
    COLOR_CRYSTAL_GLOW = (0, 180, 180)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_BEAM_START = (100, 200, 255)
    COLOR_BEAM_END = (255, 50, 50)
    COLOR_AIM_LINE = (255, 255, 255)


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_game_over = pygame.font.SysFont('Consolas', 48, bold=True)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.beam_path = []
        self.current_origin = (0, 0)
        self.aim_angle = 0.0
        self.beam_energy = 1.0
        self.mirrors = []
        self.crystals = []
        self.particles = []
        self.prev_space_held = False
        self.last_reward = 0.0
        
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.beam_energy = 1.0
        self.aim_angle = self.np_random.uniform(0, 2 * math.pi)
        self.prev_space_held = False
        
        self.beam_path = [pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)]
        self.current_origin = self.beam_path[0]
        
        self._generate_level()
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        space_held = space_pressed == 1
        
        reward = -0.001 # Small time penalty

        # --- Handle Actions ---
        if movement == 3: # Rotate left
            self.aim_angle -= self.ROTATION_SPEED
        elif movement == 4: # Rotate right
            self.aim_angle += self.ROTATION_SPEED
        self.aim_angle %= (2 * math.pi)

        fire_beam_action = space_held and not self.prev_space_held
        if fire_beam_action:
            # // SFX: fire_beam
            reward += self._fire_beam()
            reward -= 0.1 # Small penalty per bounce to encourage efficiency
        self.prev_space_held = space_held
        
        # --- Update Game State ---
        self.steps += 1
        self.beam_energy -= 1.0 / self.MAX_STEPS
        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        if self.score >= self.MAX_CRYSTALS:
            # // SFX: win_game
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.beam_energy <= 0:
            # // SFX: lose_game
            reward -= 100.0
            terminated = True
            self.game_over = True
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            
        self.last_reward = reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _fire_beam(self):
        ray_origin = self.current_origin
        ray_dir = pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle))
        
        min_dist = float('inf')
        hit_point = None
        
        # Check for intersections with mirrors
        for mirror in self.mirrors:
            for i in range(4):
                p1 = mirror.clipline(ray_origin, ray_origin + ray_dir * 2000)
                if p1:
                    hit = pygame.Vector2(p1[0])
                    dist = ray_origin.distance_to(hit)
                    if 0.1 < dist < min_dist:
                        min_dist = dist
                        hit_point = hit

        # Check for intersections with screen boundaries
        # Top/Bottom
        if ray_dir.y != 0:
            for y_boundary in [0, self.HEIGHT]:
                t = (y_boundary - ray_origin.y) / ray_dir.y
                if t > 0:
                    x_hit = ray_origin.x + t * ray_dir.x
                    if 0 <= x_hit <= self.WIDTH:
                        dist = ray_origin.distance_to(pygame.Vector2(x_hit, y_boundary))
                        if 0.1 < dist < min_dist:
                            min_dist = dist
                            hit_point = pygame.Vector2(x_hit, y_boundary)
        # Left/Right
        if ray_dir.x != 0:
            for x_boundary in [0, self.WIDTH]:
                t = (x_boundary - ray_origin.x) / ray_dir.x
                if t > 0:
                    y_hit = ray_origin.y + t * ray_dir.y
                    if 0 <= y_hit <= self.HEIGHT:
                        dist = ray_origin.distance_to(pygame.Vector2(x_boundary, y_hit))
                        if 0.1 < dist < min_dist:
                            min_dist = dist
                            hit_point = pygame.Vector2(x_boundary, y_hit)

        if hit_point is None: # Should not happen with boundary checks
            hit_point = ray_origin + ray_dir * 1000

        # --- Update state after finding hit point ---
        self.beam_path.append(hit_point)
        self.current_origin = hit_point
        self.beam_energy -= self.BOUNCE_ENERGY_COST
        
        # // SFX: bounce_mirror
        self._create_particles(hit_point, self.COLOR_BEAM_START, 20)

        # --- Check for crystal collection ---
        bounce_reward = 0
        new_segment_start = self.beam_path[-2]
        new_segment_end = self.beam_path[-1]
        
        collected_indices = []
        for i, crystal in enumerate(self.crystals):
            if crystal.clipline(new_segment_start, new_segment_end):
                collected_indices.append(i)
                self.score += 1
                bounce_reward += 10.0
                # // SFX: collect_crystal
                self._create_particles(pygame.Vector2(crystal.center), self.COLOR_CRYSTAL, 40)
        
        # Remove collected crystals safely
        for i in sorted(collected_indices, reverse=True):
            del self.crystals[i]
            
        return bounce_reward
    
    def _generate_level(self):
        self.mirrors = []
        self.crystals = []
        
        # Generate mirrors
        for _ in range(self.np_random.integers(5, 9)):
            w = self.np_random.integers(50, 150)
            h = 10
            x = self.np_random.integers(50, self.WIDTH - 50 - w)
            y = self.np_random.integers(50, self.HEIGHT - 50 - h)
            angle = self.np_random.uniform(0, 180)
            
            # Simple rotated rect approximation - not perfect but visually sufficient
            # For physics, we'll use an unrotated rect that contains it
            center_x, center_y = x + w / 2, y + h / 2
            points = [
                (x, y), (x + w, y), (x + w, y + h), (x, y + h)
            ]
            
            rotated_points = []
            for px, py in points:
                dx = px - center_x
                dy = py - center_y
                new_x = dx * math.cos(math.radians(angle)) - dy * math.sin(math.radians(angle)) + center_x
                new_y = dx * math.sin(math.radians(angle)) + dy * math.cos(math.radians(angle)) + center_y
                rotated_points.append((new_x, new_y))

            # Use an axis-aligned bounding box for collision for simplicity
            min_x = min(p[0] for p in rotated_points)
            max_x = max(p[0] for p in rotated_points)
            min_y = min(p[1] for p in rotated_points)
            max_y = max(p[1] for p in rotated_points)
            
            new_mirror = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
            self.mirrors.append(new_mirror)

        # Generate crystals
        occupied_rects = self.mirrors[:]
        for _ in range(self.MAX_CRYSTALS):
            while True:
                size = 15
                x = self.np_random.integers(20, self.WIDTH - 20 - size)
                y = self.np_random.integers(20, self.HEIGHT - 20 - size)
                new_crystal = pygame.Rect(x, y, size, size)
                if new_crystal.collidelist(occupied_rects) == -1:
                    self.crystals.append(new_crystal)
                    occupied_rects.append(new_crystal)
                    break

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "beam_energy": self.beam_energy,
            "bounces": max(0, len(self.beam_path) - 1),
        }

    def _render_game(self):
        # Draw mirrors
        for mirror in self.mirrors:
            pygame.draw.rect(self.screen, self.COLOR_MIRROR_GLOW, mirror.inflate(6, 6), border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_MIRROR, mirror, border_radius=3)
            
        # Draw crystals
        for crystal in self.crystals:
            glow_rect = crystal.inflate(8, 8)
            pygame.gfxdraw.box(self.screen, glow_rect, (*self.COLOR_CRYSTAL_GLOW, 100))
            pygame.draw.rect(self.screen, self.COLOR_CRYSTAL, crystal)

        # Draw beam path
        if len(self.beam_path) > 1:
            num_segments = len(self.beam_path) - 1
            for i in range(num_segments):
                start_pos = self.beam_path[i]
                end_pos = self.beam_path[i+1]
                
                # Interpolate color and width based on energy and age
                segment_progress = (i + 1) / num_segments
                energy_factor = max(0, self.beam_energy)
                final_progress = 1 - (segment_progress * 0.3 + (1 - energy_factor) * 0.7)

                color = self._lerp_color(self.COLOR_BEAM_END, self.COLOR_BEAM_START, final_progress)
                width = int(2 + 6 * final_progress)
                
                # Glow effect
                pygame.draw.line(self.screen, (*color, 50), start_pos, end_pos, width + 4)
                pygame.draw.line(self.screen, color, start_pos, end_pos, width)
        
        # Draw aiming line if not game over
        if not self.game_over:
            start_aim = self.current_origin
            end_aim = start_aim + pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * 1000
            self._draw_dashed_line(start_aim, end_aim)
        
        # Draw particles
        for p in self.particles:
            p_color = self._lerp_color(self.COLOR_BG, p['color'], p['life'] / p['max_life'])
            pygame.draw.circle(self.screen, p_color, p['pos'], int(p['radius'] * (p['life'] / p['max_life'])))

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"CRYSTALS: {self.score}/{self.MAX_CRYSTALS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Energy/Time display
        energy_percent = max(0, self.beam_energy) * 100
        time_color = self._lerp_color(self.COLOR_BEAM_END, self.COLOR_UI_TEXT, max(0, self.beam_energy))
        time_text = self.font_ui.render(f"ENERGY: {energy_percent:.1f}%", True, time_color)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.MAX_CRYSTALS:
                msg = "VICTORY"
                color = self.COLOR_CRYSTAL
            else:
                msg = "BEAM FADED"
                color = self.COLOR_BEAM_END
                
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)

    def _lerp_color(self, c1, c2, t):
        t = max(0, min(1, t))
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )

    def _draw_dashed_line(self, start_pos, end_pos, dash_length=10, gap_length=5):
        line_vec = pygame.Vector2(end_pos) - pygame.Vector2(start_pos)
        distance = line_vec.length()
        if distance == 0: return
        
        direction = line_vec.normalize()
        
        current_pos = pygame.Vector2(start_pos)
        traveled = 0
        drawing = True
        
        while traveled < distance:
            if drawing:
                segment_end = current_pos + direction * dash_length
                if (traveled + dash_length) > distance:
                    segment_end = end_pos
                pygame.draw.line(self.screen, (*self.COLOR_AIM_LINE, 150), current_pos, segment_end, 1)
                traveled += dash_length
            else:
                traveled += gap_length
            current_pos += direction * (dash_length if drawing else gap_length)
            drawing = not drawing

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'radius': self.np_random.uniform(2, 6),
                'color': color,
                'life': self.np_random.integers(15, 30),
                'max_life': 30
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a graphical display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print("--- Manual Control ---")
    print("Left/Right Arrows: Aim")
    print("Spacebar: Fire Beam")
    print("Q: Quit")
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Beam Bounce Environment")
    clock = pygame.time.Clock()
    
    while not done:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True
        
        clock.tick(GameEnv.FPS)
        
    print(f"Game Over. Final Info: {info}")
    env.close()