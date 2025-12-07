import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:17:55.114965
# Source Brief: brief_02689.md
# Brief Index: 2689
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the agent controls two platforms to deflect falling blocks.
    The goal is to survive for 60 seconds with a collision rate below 50%.
    A "collision" in this context refers to a "bad" deflection, where the speed
    difference between the block and the platform is too high.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    game_description = (
        "Control two vertical platforms to deflect falling blocks. Match block speeds for "
        "good deflections and survive for 60 seconds with a low bad collision rate."
    )
    user_guide = (
        "Controls: Use ↑/↓ to control the left platform's speed and ←/→ for the right. "
        "Press space/shift to toggle ramp mode for each platform."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 60 * FPS  # 60 seconds * 60 fps
    WIN_TIME = 60.0  # 60 seconds to win
    FAIL_COLLISION_RATE = 0.5

    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_TRACK = (70, 70, 90)
    COLOR_PLATFORM = (220, 220, 230)
    COLOR_PLATFORM_RAMP = (230, 180, 100)
    COLOR_BLOCK_SLOW = (80, 220, 80)
    COLOR_BLOCK_FAST = (220, 80, 80)
    COLOR_SPARK = (255, 255, 100)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_VALUE_GOOD = (100, 255, 100)
    COLOR_UI_VALUE_BAD = (255, 100, 100)

    # Physics & Gameplay
    PLATFORM_WIDTH = 80
    PLATFORM_HEIGHT = 15
    PLATFORM_MIN_SPEED = 0
    PLATFORM_MAX_SPEED = 15
    PLATFORM_SPEED_CHANGE = 1.0 # Units per second
    TRACK_1_X = SCREEN_WIDTH // 3
    TRACK_2_X = SCREEN_WIDTH * 2 // 3
    BLOCK_SIZE = 20
    BLOCK_MIN_SPEED_Y = 2.0
    BLOCK_MAX_SPEED_Y = 8.0
    INITIAL_SPAWN_RATE = 0.5  # Blocks per second
    SPAWN_RATE_INCREASE = 0.01 # Per second
    COLLISION_SPEED_TOLERANCE = 2.0 # Max speed diff for a "good" collision
    RAMP_DEFLECTION_VX = 5.0 # Horizontal speed imparted by ramp

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_timer = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_time = 0.0
        self.platforms = []
        self.blocks = []
        self.particles = []
        self.total_deflections = 0
        self.bad_collisions = 0
        self.collision_rate = 0.0
        self.spawn_timer = 0.0
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_reward_check_time = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_time = 0.0
        
        self.platforms = [
            self._create_platform(self.TRACK_1_X),
            self._create_platform(self.TRACK_2_X)
        ]
        self.blocks = []
        self.particles = []

        self.total_deflections = 0
        self.bad_collisions = 0
        self.collision_rate = 0.0
        
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE
        self.spawn_timer = 1.0 / self.current_spawn_rate

        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_reward_check_time = 0.0

        return self._get_observation(), self._get_info()

    def _create_platform(self, x_pos):
        return {
            "rect": pygame.Rect(x_pos - self.PLATFORM_WIDTH // 2, self.SCREEN_HEIGHT - 50, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT),
            "speed": 0.0, # in pixels per second
            "is_ramp": False,
        }

    def step(self, action):
        dt = self.clock.tick(self.FPS) / 1000.0
        self.steps += 1
        self.game_time += dt
        
        reward = 0
        
        # --- Handle Actions ---
        self._handle_input(action, dt)
        
        # --- Update Game Logic ---
        self._update_platforms(dt)
        self._update_spawner(dt)
        self._update_blocks(dt)
        collision_penalty = self._handle_collisions()
        self._update_particles(dt)

        # --- Calculate Reward ---
        # Continuous survival reward
        if self.game_time - self.last_reward_check_time >= 0.1:
            prev_collision_rate = self.collision_rate
            self._update_collision_rate()
            if self.collision_rate <= prev_collision_rate:
                 reward += 0.01 # Scaled down from brief to balance with collision penalty
            self.last_reward_check_time = self.game_time
        
        # Collision penalty is applied in _handle_collisions
        reward += collision_penalty
        self.score += reward

        # --- Check Termination ---
        terminated = self._check_termination()
        
        # Terminal rewards
        if terminated:
            if self.game_time >= self.WIN_TIME and self.collision_rate < self.FAIL_COLLISION_RATE:
                reward += 100
                self.score += 100
            elif self.collision_rate >= self.FAIL_COLLISION_RATE:
                reward -= 100
                self.score -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action, dt):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Platform 1 speed
        if movement == 1: # Up
            self.platforms[0]['speed'] += self.PLATFORM_SPEED_CHANGE
        elif movement == 2: # Down
            self.platforms[0]['speed'] -= self.PLATFORM_SPEED_CHANGE

        # Platform 2 speed
        if movement == 3: # Left
            self.platforms[1]['speed'] += self.PLATFORM_SPEED_CHANGE
        elif movement == 4: # Right
            self.platforms[1]['speed'] -= self.PLATFORM_SPEED_CHANGE
            
        # Clamp speeds
        self.platforms[0]['speed'] = max(self.PLATFORM_MIN_SPEED, min(self.PLATFORM_MAX_SPEED, self.platforms[0]['speed']))
        self.platforms[1]['speed'] = max(self.PLATFORM_MIN_SPEED, min(self.PLATFORM_MAX_SPEED, self.platforms[1]['speed']))

        # Toggle ramps on press (not hold)
        if space_held and not self.prev_space_held:
            self.platforms[0]['is_ramp'] = not self.platforms[0]['is_ramp']
        if shift_held and not self.prev_shift_held:
            self.platforms[1]['is_ramp'] = not self.platforms[1]['is_ramp']

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_platforms(self, dt):
        for p in self.platforms:
            # Note: speed is in units/sec, but platforms move up/down on screen
            # So a positive speed means moving UP (decreasing y)
            p['rect'].y -= p['speed'] * 10 * dt # Multiply by 10 for better visual scaling
            # Boundary checks
            p['rect'].top = max(0, p['rect'].top)
            p['rect'].bottom = min(self.SCREEN_HEIGHT, p['rect'].bottom)

    def _update_spawner(self, dt):
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE + self.game_time * self.SPAWN_RATE_INCREASE
        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            self._spawn_block()
            self.spawn_timer += 1.0 / self.current_spawn_rate

    def _spawn_block(self):
        track_x = random.choice([self.TRACK_1_X, self.TRACK_2_X])
        speed_y = self.np_random.uniform(self.BLOCK_MIN_SPEED_Y, self.BLOCK_MAX_SPEED_Y)
        
        speed_ratio = (speed_y - self.BLOCK_MIN_SPEED_Y) / (self.BLOCK_MAX_SPEED_Y - self.BLOCK_MIN_SPEED_Y)
        color = self._interpolate_color(self.COLOR_BLOCK_SLOW, self.COLOR_BLOCK_FAST, speed_ratio)

        self.blocks.append({
            "rect": pygame.Rect(track_x - self.BLOCK_SIZE // 2, -self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE),
            "vx": 0.0,
            "vy": speed_y,
            "color": color
        })

    def _update_blocks(self, dt):
        for b in self.blocks[:]:
            b['rect'].x += b['vx'] * 10 * dt
            b['rect'].y += b['vy'] * 10 * dt
            if b['rect'].top > self.SCREEN_HEIGHT:
                self.blocks.remove(b)

    def _handle_collisions(self):
        reward_penalty = 0
        for b in self.blocks[:]:
            for p in self.platforms:
                if b['rect'].colliderect(p['rect']):
                    self.total_deflections += 1
                    speed_diff = abs(b['vy'] - p['speed'])

                    if speed_diff > self.COLLISION_SPEED_TOLERANCE:
                        self.bad_collisions += 1
                        reward_penalty -= 1 # Event-based penalty
                    
                    self._create_sparks(b['rect'].midbottom)

                    # Deflection physics
                    if p['is_ramp']:
                        # Deflect horizontally
                        direction = 1 if b['rect'].centerx < self.SCREEN_WIDTH / 2 else -1
                        b['vx'] = self.RAMP_DEFLECTION_VX * direction
                        b['vy'] *= 0.5 # Lose some vertical speed
                    else:
                        # Bounce vertically
                        b['vy'] = -b['vy'] * 0.8 # Dampen bounce
                    
                    # Move block out of platform to prevent re-collision
                    b['rect'].bottom = p['rect'].top 
        
        return reward_penalty

    def _update_particles(self, dt):
        for particle in self.particles[:]:
            particle['pos'][0] += particle['vel'][0] * dt
            particle['pos'][1] += particle['vel'][1] * dt
            particle['life'] -= dt
            if particle['life'] <= 0:
                self.particles.remove(particle)

    def _update_collision_rate(self):
        if self.total_deflections > 0:
            self.collision_rate = self.bad_collisions / self.total_deflections
        else:
            self.collision_rate = 0.0
    
    def _check_termination(self):
        if self.collision_rate >= self.FAIL_COLLISION_RATE and self.total_deflections > 5: # Add a grace period
            return True
        if self.game_time >= self.WIN_TIME:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        # Pygame surfaces are (width, height), but our observation space is (height, width)
        # So we need to transpose the array from pygame.
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "game_time": self.game_time,
            "collision_rate": self.collision_rate,
            "total_deflections": self.total_deflections,
            "bad_collisions": self.bad_collisions,
        }
        
    def _render_game(self):
        # Draw tracks
        pygame.draw.line(self.screen, self.COLOR_TRACK, (self.TRACK_1_X, 0), (self.TRACK_1_X, self.SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (self.TRACK_2_X, 0), (self.TRACK_2_X, self.SCREEN_HEIGHT), 2)
        
        # Draw platforms
        for p in self.platforms:
            color = self.COLOR_PLATFORM_RAMP if p['is_ramp'] else self.COLOR_PLATFORM
            if p['is_ramp']:
                # Draw as a trapezoid
                rect = p['rect']
                points = [
                    (rect.left, rect.bottom), (rect.right, rect.bottom),
                    (rect.right - 10, rect.top), (rect.left + 10, rect.top)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            else:
                pygame.draw.rect(self.screen, color, p['rect'], border_radius=3)

        # Draw blocks
        for b in self.blocks:
            pygame.draw.rect(self.screen, b['color'], b['rect'], border_radius=2)

        # Draw particles
        for particle in self.particles:
            size = max(0, int(particle['life'] * particle['start_size']))
            if size > 0:
                pygame.draw.circle(self.screen, self.COLOR_SPARK, [int(p) for p in particle['pos']], size)

    def _render_ui(self):
        # Collision Rate
        self._update_collision_rate()
        rate_text = f"FAIL RATE: {self.collision_rate:.1%}"
        rate_color = self.COLOR_UI_VALUE_BAD if self.collision_rate >= self.FAIL_COLLISION_RATE else self.COLOR_UI_VALUE_GOOD
        text_surf = self.font_ui.render(rate_text, True, rate_color)
        self.screen.blit(text_surf, (10, 10))
        
        # Timer
        time_left = max(0, self.WIN_TIME - self.game_time)
        timer_text = f"{time_left:.1f}"
        self._draw_text(timer_text, (self.SCREEN_WIDTH - 80, 10), self.COLOR_UI_TEXT, font=self.font_timer)

        # Platform speeds
        for i, p in enumerate(self.platforms):
            speed_text = f"SPD: {p['speed']:.1f}"
            pos = (p['rect'].centerx, p['rect'].top - 20)
            self._draw_text(speed_text, pos, self.COLOR_UI_TEXT, center=True)

    def _draw_text(self, text, pos, color, font=None, center=False):
        if font is None:
            font = self.font_ui
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _create_sparks(self, pos, count=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(50, 150)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.uniform(0.2, 0.5),
                'start_size': self.np_random.integers(1, 5)
            })

    def _interpolate_color(self, color1, color2, factor):
        r = int(color1[0] + (color2[0] - color1[0]) * factor)
        g = int(color1[1] + (color2[1] - color1[1]) * factor)
        b = int(color1[2] + (color2[2] - color1[2]) * factor)
        return (r, g, b)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in the headless evaluation environment
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Deflector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print(f"Final Info: {info}")
            # Reset on termination for continuous play
            obs, info = env.reset()
            total_reward = 0

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()