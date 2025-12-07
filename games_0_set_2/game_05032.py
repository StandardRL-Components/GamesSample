
# Generated: 2025-08-28T03:46:21.899775
# Source Brief: brief_05032.md
# Brief Index: 5032

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Press Space to jump over the neon obstacles. Time your jumps to the rhythm!"
    )

    game_description = (
        "A minimalist rhythm-runner. Jump over obstacles in a neon world, "
        "synchronizing your actions to the beat to achieve a high score."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    GROUND_Y = 320
    PLAYER_X = 150
    JUMP_VELOCITY = 15
    GRAVITY = 1.0

    # --- Colors ---
    COLOR_BG = (10, 5, 25)
    COLOR_ROAD = (25, 15, 50)
    COLOR_ROAD_LINES = (50, 40, 90)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    OBSTACLE_COLORS = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
    ]
    COLOR_SUCCESS = (100, 255, 100)
    COLOR_FAIL = (255, 100, 100)
    COLOR_TEXT = (240, 240, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 40, bold=True)

        # These will be initialized in reset()
        self.steps = 0
        self.game_over = False
        self.player_y = 0
        self.player_vy = 0
        self.is_jumping = False
        self.prev_space_held = False
        self.obstacles = []
        self.particles = []
        self.obstacle_speed = 0
        self.obstacle_spawn_timer = 0
        self.jumps_hit = 0
        self.obstacles_passed = 0
        self.hit_50_pct_bonus = False
        self.hit_80_pct_bonus = False
        self.final_message = ""
        self.screen_flash = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.game_over = False
        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.is_jumping = False
        self.prev_space_held = False
        
        self.obstacles = []
        self.particles = []
        
        self.obstacle_speed = 5.0
        self.obstacle_spawn_timer = 30 # Spawn first obstacle quickly
        
        self.jumps_hit = 0
        self.obstacles_passed = 0
        self.hit_50_pct_bonus = False
        self.hit_80_pct_bonus = False
        self.final_message = ""
        self.screen_flash = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(30)
        self.steps += 1
        reward = 0

        space_held = action[1] == 1
        
        self._handle_input(space_held)
        self._update_player()
        
        obstacle_reward = self._update_obstacles()
        reward += obstacle_reward
        
        self._update_particles()
        self._update_difficulty()
        self._spawn_obstacles()

        accuracy = (self.jumps_hit / self.obstacles_passed) if self.obstacles_passed > 0 else 1.0
        
        if not self.hit_50_pct_bonus and accuracy >= 0.5 and self.obstacles_passed > 5:
            reward += 5
            self.hit_50_pct_bonus = True
        
        if not self.hit_80_pct_bonus and accuracy >= 0.8 and self.obstacles_passed > 10:
            reward += 10
            self.hit_80_pct_bonus = True

        terminated = self._check_termination(accuracy)
        if terminated:
            self.game_over = True
            if self.final_message == "VICTORY!":
                reward += 100
            else:
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, space_held):
        if space_held and not self.prev_space_held and not self.is_jumping:
            self.is_jumping = True
            self.player_vy = -self.JUMP_VELOCITY
            # SFX: Jump
        self.prev_space_held = space_held

    def _update_player(self):
        if self.is_jumping:
            self.player_y += self.player_vy
            self.player_vy += self.GRAVITY
            if self.player_y >= self.GROUND_Y:
                self.player_y = self.GROUND_Y
                self.is_jumping = False
                self.player_vy = 0
                # SFX: Land
                self._create_particles(self.PLAYER_X, self.GROUND_Y, self.COLOR_PLAYER, 5, is_landing=True)

    def _update_obstacles(self):
        reward = 0
        for obs in self.obstacles:
            obs['x'] -= self.obstacle_speed
            
            # Check for scoring
            if not obs['scored'] and obs['x'] + obs['w'] < self.PLAYER_X:
                obs['scored'] = True
                self.obstacles_passed += 1
                
                player_rect = pygame.Rect(self.PLAYER_X - 10, self.player_y - 20, 20, 20)
                obs_rect = pygame.Rect(obs['x'], obs['y']-obs['h'], obs['w'], obs['h'])

                # A successful jump is if the player is airborne when the obstacle passes
                if self.is_jumping and player_rect.top < obs_rect.top:
                    self.jumps_hit += 1
                    reward += 1
                    # SFX: Success
                    self._create_particles(self.PLAYER_X, obs['y']-obs['h']/2, obs['color'], 20)
                else:
                    reward -= 1
                    self.screen_flash = 10 # Frames to flash
                    # SFX: Fail
                    self._create_particles(self.PLAYER_X, self.player_y, self.COLOR_FAIL, 15)
        
        # Cleanup off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['x'] + obs['w'] > 0]
        return reward

    def _spawn_obstacles(self):
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            w = self.np_random.integers(20, 40)
            h = self.np_random.integers(40, 100)
            self.obstacles.append({
                'x': self.SCREEN_WIDTH + 50,
                'y': self.GROUND_Y,
                'w': w,
                'h': h,
                'color': random.choice(self.OBSTACLE_COLORS),
                'scored': False
            })
            # Reset timer based on current speed
            min_gap = int(self.SCREEN_WIDTH / self.obstacle_speed * 0.5)
            max_gap = int(self.SCREEN_WIDTH / self.obstacle_speed * 1.0)
            self.obstacle_spawn_timer = self.np_random.integers(min_gap, max_gap)

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.obstacle_speed = min(15.0, self.obstacle_speed + 0.05)

    def _check_termination(self, accuracy):
        if self.steps >= self.MAX_STEPS:
            if accuracy >= 0.8:
                self.final_message = "VICTORY!"
            else:
                self.final_message = "TIME'S UP!"
            return True
            
        # Check for loss condition after a grace period
        if self.obstacles_passed > 10 and (self.obstacles_passed - self.jumps_hit) / self.obstacles_passed > 0.2:
            self.final_message = "ACCURACY TOO LOW"
            return True
            
        return False

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Particle gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_particles(self, x, y, color, count, is_landing=False):
        for _ in range(count):
            if is_landing:
                angle = self.np_random.uniform(math.pi, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 6)
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self._render_background()
        self._render_obstacles()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        
        # Rhythmic pulse
        pulse_progress = (self.steps % 45) / 45.0
        pulse_alpha = int((1 - pulse_progress) * 30)
        pulse_radius = int(pulse_progress * self.SCREEN_WIDTH)
        pulse_color = self.COLOR_ROAD_LINES
        
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.aacircle(s, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, pulse_radius, pulse_color + (pulse_alpha,))
        self.screen.blit(s, (0,0))
        
        # Road
        road_rect = pygame.Rect(0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_ROAD, road_rect)
        pygame.draw.line(self.screen, self.COLOR_ROAD_LINES, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 2)
        
        # Perspective lines
        vanishing_point = (self.SCREEN_WIDTH / 2, self.GROUND_Y - 150)
        for i in range(5):
            y = self.GROUND_Y + i * 20
            x_left = (vanishing_point[0] * (y - vanishing_point[1])) / (self.SCREEN_HEIGHT - vanishing_point[1])
            x_right = self.SCREEN_WIDTH - x_left
            if y < self.SCREEN_HEIGHT:
                pygame.draw.line(self.screen, self.COLOR_ROAD_LINES, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_obstacles(self):
        for obs in self.obstacles:
            rect = pygame.Rect(int(obs['x']), int(obs['y'] - obs['h']), int(obs['w']), int(obs['h']))
            pygame.draw.rect(self.screen, obs['color'], rect)
            pygame.gfxdraw.rectangle(self.screen, rect, self.COLOR_PLAYER + (150,))

    def _render_player(self):
        player_pos = (int(self.PLAYER_X), int(self.player_y))
        
        # Glow effect
        for i in range(10, 0, -1):
            alpha = 150 * (1 - (i / 10))
            pygame.gfxdraw.filled_circle(self.screen, player_pos[0], player_pos[1] - 10, 10 + i, (*self.COLOR_PLAYER_GLOW, int(alpha)))
            
        # Player Icon (a simple triangle)
        points = [
            (player_pos[0], player_pos[1] - 20),
            (player_pos[0] - 10, player_pos[1]),
            (player_pos[0] + 10, player_pos[1]),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = p['color'] + (alpha,)
            pos = (int(p['x']), int(p['y']))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), color)

    def _render_ui(self):
        # Fail flash
        if self.screen_flash > 0:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            s.set_alpha(self.screen_flash * 15)
            s.fill(self.COLOR_FAIL)
            self.screen.blit(s, (0,0))
            self.screen_flash -= 1

        # Accuracy display
        accuracy = (self.jumps_hit / self.obstacles_passed * 100) if self.obstacles_passed > 0 else 100.0
        acc_text = f"ACCURACY: {accuracy:.1f}%"
        text_surface = self.font_ui.render(acc_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        # Steps display
        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        text_surface = self.font_ui.render(steps_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 10, 10))
        
        # Game Over message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            s.set_alpha(180)
            s.fill((0,0,0))
            self.screen.blit(s, (0,0))

            msg_surface = self.font_msg.render(self.final_message, True, self.COLOR_PLAYER)
            msg_rect = msg_surface.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surface, msg_rect)

    def _get_info(self):
        accuracy = (self.jumps_hit / self.obstacles_passed) if self.obstacles_passed > 0 else 1.0
        return {
            "steps": self.steps,
            "jumps_hit": self.jumps_hit,
            "obstacles_passed": self.obstacles_passed,
            "accuracy": accuracy
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test testable assertions from brief
        self.reset()
        initial_speed = self.obstacle_speed
        for _ in range(201):
            self.step(self.action_space.sample())
        assert self.obstacle_speed > initial_speed, "Obstacle speed did not increase after 200 steps"
        
        print("✓ Implementation validated successfully")