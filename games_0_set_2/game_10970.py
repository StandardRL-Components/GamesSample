import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:22:34.033879
# Source Brief: brief_00970.md
# Brief Index: 970
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a high-speed train.
    The goal is to navigate procedural tracks, collect items, and avoid
    derailing or crashing across 5 levels within a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a high-speed train on a procedural track. "
        "Collect items and maintain speed without derailing to complete all the levels."
    )
    user_guide = "Controls: Use ↑ to accelerate and ↓ to decelerate. Stay on the track and avoid obstacles."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (30, 50, 80)
    COLOR_TRACK = (100, 100, 110)
    COLOR_TRACK_TIES = (70, 70, 80)
    COLOR_TRAIN = (255, 60, 60)
    COLOR_TRAIN_GLOW = (255, 100, 100)
    COLOR_COIN = (255, 215, 0)
    COLOR_BOOST = (0, 150, 255)
    COLOR_OBSTACLE = (10, 10, 10)
    COLOR_TEXT = (240, 240, 240)
    COLOR_UI_BG = (10, 20, 40, 180)

    # Game Parameters
    MAX_LEVELS = 5
    LEVEL_TIME_LIMIT = 120  # seconds
    MAX_STEPS_PER_LEVEL = LEVEL_TIME_LIMIT * 30 # Assuming 30 FPS
    
    TRAIN_ACCELERATION = 0.05
    TRAIN_DECELERATION = 0.1
    FRICTION = 0.995
    MIN_SPEED = 0.5
    MAX_SPEED = 10.0
    DERAIL_SPEED_FACTOR = 150.0 # Lower is harder

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Pre-render background for performance
        self._background = self._create_gradient_background()

        # Initialize state variables (will be properly set in reset)
        self.level = 1
        self.steps = 0
        self.level_steps = 0
        self.score = 0
        self.game_over = False
        self.train_pos_world = 0.0
        self.train_speed = 0.0
        self.train_y = 0.0
        self.train_angle = 0.0
        self.camera_offset = 0.0
        self.track_nodes = []
        self.track_length = 0
        self.items = []
        self.obstacles = []
        self.particles = []
        self.coins_collected = 0
        self.boosts_collected = 0
        
        # self.reset() # This is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.train_speed = self.MIN_SPEED
        self.particles = []
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.level_steps += 1
        reward = 0

        # 1. Handle Actions
        movement = action[0]
        if movement == 1:  # Up: Accelerate
            self.train_speed += self.TRAIN_ACCELERATION
            # Placeholder for sound effect: pygame.mixer.Sound('accelerate.wav').play()
        elif movement == 2:  # Down: Decelerate
            self.train_speed -= self.TRAIN_DECELERATION
            reward -= 0.05 # Small penalty for braking
        
        # 2. Update Physics
        self.train_speed *= self.FRICTION
        self.train_speed = np.clip(self.train_speed, self.MIN_SPEED, self.MAX_SPEED)
        self.train_pos_world += self.train_speed
        
        # Add reward for moving forward
        if self.train_speed > self.MIN_SPEED:
            reward += 0.01 * (self.train_speed / self.MAX_SPEED)

        # 3. Update Train Position on Track
        self.train_y, self.train_angle, curvature = self._get_track_info(self.train_pos_world)
        self.camera_offset = self.train_pos_world - self.SCREEN_WIDTH * 0.2

        # 4. Check for Collisions and Derailing
        reward += self._check_collisions()
        
        derail_threshold = self.DERAIL_SPEED_FACTOR / (1 + abs(curvature))
        if self.train_speed > derail_threshold and self.train_speed > 3.0:
            self.game_over = True
            reward = -100
            # Placeholder for sound effect: pygame.mixer.Sound('crash.wav').play()

        # 5. Check for Level Completion
        if self.train_pos_world >= self.track_length:
            # Placeholder for sound effect: pygame.mixer.Sound('level_complete.wav').play()
            self.score += 10
            reward += 10
            self.level += 1
            if self.level > self.MAX_LEVELS:
                self.game_over = True
                self.score += 100
                reward += 100
            else:
                self._generate_level()

        # 6. Check Termination Conditions
        if self.level_steps >= self.MAX_STEPS_PER_LEVEL:
            self.game_over = True
            reward = -100 # Penalize for running out of time
        
        terminated = self.game_over
        truncated = False
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_level(self):
        self.train_pos_world = 0.0
        self.level_steps = 0
        self.coins_collected = 0
        self.boosts_collected = 0
        self.items = []
        self.obstacles = []

        # Procedurally generate track
        self.track_length = 3000 + self.level * 1000
        y_offset = self.SCREEN_HEIGHT / 2
        amplitude = 30 + self.level * 15
        frequency = 2 + self.level * 0.5
        
        self.track_nodes = []
        for x in range(0, int(self.track_length) + 100, 20):
            y = y_offset + amplitude * math.sin(x / self.track_length * frequency * 2 * math.pi)
            self.track_nodes.append((x, y))

        # Place items and obstacles
        obstacle_density = 0.02 + 0.02 * self.level
        coin_density = 0.1
        boost_density = 0.03

        for i in range(1, len(self.track_nodes) - 1):
            x, y = self.track_nodes[i]
            rand = self.np_random.random()
            
            if rand < obstacle_density:
                self.obstacles.append({'pos': (x, y), 'size': 15})
            elif rand < obstacle_density + coin_density:
                self.items.append({'type': 'coin', 'pos': (x, y), 'size': 8})
            elif rand < obstacle_density + coin_density + boost_density:
                self.items.append({'type': 'boost', 'pos': (x, y), 'size': 10})

    def _get_track_info(self, x_pos):
        # Find the segment the train is on
        segment_idx = int(x_pos // 20)
        if segment_idx >= len(self.track_nodes) - 2:
            segment_idx = len(self.track_nodes) - 3

        p1 = self.track_nodes[segment_idx]
        p2 = self.track_nodes[segment_idx + 1]
        
        # Interpolate Y position
        interp = (x_pos % 20) / 20
        y_pos = p1[1] + interp * (p2[1] - p1[1])
        
        # Calculate angle
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

        # Approximate curvature
        p3 = self.track_nodes[segment_idx + 2]
        angle2 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
        curvature = angle2 - angle
        
        return y_pos, angle, curvature

    def _check_collisions(self):
        reward = 0
        train_hitbox = pygame.Rect(self.SCREEN_WIDTH * 0.2 - 10, self.train_y - 10, 20, 20)
        
        # Item collisions
        for item in self.items[:]:
            item_x_screen = item['pos'][0] - self.camera_offset
            if abs(item_x_screen - train_hitbox.centerx) < 50: # Broad-phase check
                item_rect = pygame.Rect(item_x_screen - item['size'], item['pos'][1] - item['size'], item['size']*2, item['size']*2)
                if train_hitbox.colliderect(item_rect):
                    if item['type'] == 'coin':
                        self.score += 1
                        reward += 1
                        self.coins_collected += 1
                        self._create_particles(item['pos'], self.COLOR_COIN, 10)
                        # Placeholder: pygame.mixer.Sound('coin.wav').play()
                    elif item['type'] == 'boost':
                        self.score += 2
                        reward += 2
                        self.train_speed = min(self.MAX_SPEED, self.train_speed + 2.0)
                        self.boosts_collected += 1
                        self._create_particles(item['pos'], self.COLOR_BOOST, 20)
                        # Placeholder: pygame.mixer.Sound('boost.wav').play()
                    self.items.remove(item)

        # Obstacle collisions
        for obs in self.obstacles:
            obs_x_screen = obs['pos'][0] - self.camera_offset
            if abs(obs_x_screen - train_hitbox.centerx) < 50:
                obs_rect = pygame.Rect(obs_x_screen - obs['size'], obs['pos'][1] - obs['size'], obs['size']*2, obs['size']*2)
                if train_hitbox.colliderect(obs_rect):
                    self.game_over = True
                    reward = -100
                    # Placeholder: pygame.mixer.Sound('crash.wav').play()
                    break
        return reward

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'lifespan': self.np_random.uniform(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.blit(self._background, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Update and draw particles first (behind everything)
        self._update_particles()
        for p in self.particles:
            pos_on_screen = (int(p['pos'][0] - self.camera_offset), int(p['pos'][1]))
            size = int(p['lifespan'] / 6)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], pos_on_screen, size)

        # Draw track
        for i in range(len(self.track_nodes) - 1):
            p1 = self.track_nodes[i]
            p2 = self.track_nodes[i+1]
            p1_screen = (p1[0] - self.camera_offset, p1[1])
            p2_screen = (p2[0] - self.camera_offset, p2[1])

            if max(p1_screen[0], p2_screen[0]) > 0 and min(p1_screen[0], p2_screen[0]) < self.SCREEN_WIDTH:
                # Draw track ties
                if i % 4 == 0:
                    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    perp_angle = angle + math.pi / 2
                    tie_w = 15
                    t1 = (p1_screen[0] + math.cos(perp_angle) * tie_w, p1_screen[1] + math.sin(perp_angle) * tie_w)
                    t2 = (p1_screen[0] - math.cos(perp_angle) * tie_w, p1_screen[1] - math.sin(perp_angle) * tie_w)
                    pygame.draw.aaline(self.screen, self.COLOR_TRACK_TIES, t1, t2, 2)
                # Draw rails
                pygame.draw.aaline(self.screen, self.COLOR_TRACK, (p1_screen[0], p1_screen[1]-5), (p2_screen[0], p2_screen[1]-5), 2)
                pygame.draw.aaline(self.screen, self.COLOR_TRACK, (p1_screen[0], p1_screen[1]+5), (p2_screen[0], p2_screen[1]+5), 2)

        # Draw items and obstacles
        for item in self.items:
            pos_on_screen = (int(item['pos'][0] - self.camera_offset), int(item['pos'][1]))
            if 0 < pos_on_screen[0] < self.SCREEN_WIDTH:
                color = self.COLOR_COIN if item['type'] == 'coin' else self.COLOR_BOOST
                pygame.gfxdraw.filled_circle(self.screen, pos_on_screen[0], pos_on_screen[1], item['size'], color)
                pygame.gfxdraw.aacircle(self.screen, pos_on_screen[0], pos_on_screen[1], item['size'], (255,255,255))
        
        for obs in self.obstacles:
            pos_on_screen = (int(obs['pos'][0] - self.camera_offset), int(obs['pos'][1]))
            if 0 < pos_on_screen[0] < self.SCREEN_WIDTH:
                pygame.gfxdraw.filled_circle(self.screen, pos_on_screen[0], pos_on_screen[1], obs['size'], self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, pos_on_screen[0], pos_on_screen[1], obs['size'], (50,50,50))

        # Draw train
        train_screen_x = int(self.SCREEN_WIDTH * 0.2)
        train_screen_y = int(self.train_y)
        
        # Rotated train body
        train_len = 20
        train_width = 8
        points = [
            (-train_len/2, -train_width/2), (train_len/2, -train_width/2),
            (train_len/2, train_width/2), (-train_len/2, train_width/2)
        ]
        rotated_points = []
        for x, y in points:
            new_x = x * math.cos(self.train_angle) - y * math.sin(self.train_angle) + train_screen_x
            new_y = x * math.sin(self.train_angle) + y * math.cos(self.train_angle) + train_screen_y
            rotated_points.append((new_x, new_y))

        # Draw glow
        glow_color = list(self.COLOR_TRAIN_GLOW)
        glow_color.append(int(100 + 155 * (self.train_speed / self.MAX_SPEED))) # Glow intensity based on speed
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, glow_color)
        
        # Draw main body
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_TRAIN)
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, (255, 255, 255))

    def _render_ui(self):
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)

        # Top-left info
        level_text = self.font_small.render(f"LEVEL: {self.level}/{self.MAX_LEVELS}", True, self.COLOR_TEXT)
        coin_text = self.font_small.render(f"COINS: {self.coins_collected}", True, self.COLOR_COIN)
        boost_text = self.font_small.render(f"BOOSTS: {self.boosts_collected}", True, self.COLOR_BOOST)
        
        time_left = (self.MAX_STEPS_PER_LEVEL - self.level_steps) / 30
        timer_text = self.font_large.render(f"TIME: {int(time_left)}", True, self.COLOR_TEXT)
        
        ui_surf.blit(level_text, (10, 5))
        ui_surf.blit(coin_text, (10, 25))
        ui_surf.blit(boost_text, (130, 25))
        ui_surf.blit(timer_text, (250, 10))
        
        # Bottom-right info (speed)
        speed_bar_width = 150
        speed_bar_height = 20
        speed_ratio = (self.train_speed - self.MIN_SPEED) / (self.MAX_SPEED - self.MIN_SPEED)
        current_speed_width = int(speed_bar_width * speed_ratio)

        br_surf = pygame.Surface((speed_bar_width + 20, speed_bar_height + 20), pygame.SRCALPHA)
        br_surf.fill(self.COLOR_UI_BG)
        pygame.draw.rect(br_surf, (80, 80, 80), (10, 10, speed_bar_width, speed_bar_height))
        pygame.draw.rect(br_surf, self.COLOR_TRAIN, (10, 10, current_speed_width, speed_bar_height))
        speed_text = self.font_small.render("SPEED", True, self.COLOR_TEXT)
        br_surf.blit(speed_text, (speed_bar_width/2 - 20, 12))

        self.screen.blit(ui_surf, (0, 0))
        self.screen.blit(br_surf, (self.SCREEN_WIDTH - speed_bar_width - 20, self.SCREEN_HEIGHT - speed_bar_height - 20))
        
        if self.game_over:
            end_text_str = "LEVELS COMPLETE!" if self.level > self.MAX_LEVELS else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            pygame.draw.rect(self.screen, self.COLOR_UI_BG, text_rect.inflate(20, 20))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "coins": self.coins_collected,
            "boosts": self.boosts_collected,
            "train_speed": self.train_speed,
        }
        
    def _create_gradient_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage:
if __name__ == "__main__":
    # The main block is for human play and debugging.
    # It will not run in the evaluation environment.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible driver for human play
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # Controls: Arrow Up (Accelerate), Arrow Down (Decelerate)
    obs, info = env.reset()
    terminated = False
    
    # Pygame window for human play
    pygame.display.set_caption("Train Conductor")
    screen_display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action[0] = 0 # No-op initially

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN]:
                    action[0] = 0
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit to 30 FPS for human play

    env.close()
    print("Game Over!")
    print(f"Final Info: {info}")