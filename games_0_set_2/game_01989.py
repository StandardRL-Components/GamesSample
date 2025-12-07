
# Generated: 2025-08-28T03:19:33.690747
# Source Brief: brief_01989.md
# Brief Index: 1989

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Hold Space for a small jump or Shift for a large jump to clear the obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a robot through a procedurally generated obstacle course to reach the finish line as quickly as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STAGES = 3
        self.TIME_LIMIT_SECONDS = 60

        # Physics
        self.GRAVITY = 0.6
        self.JUMP_SMALL = -12
        self.JUMP_LARGE = -16
        self.GROUND_Y = self.HEIGHT - 50
        self.ROBOT_X_POS = 100

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_GROUND = (180, 180, 190)
        self.COLOR_ROBOT = (60, 160, 255)
        self.COLOR_ROBOT_GLOW = (120, 200, 255)
        self.COLOR_OBSTACLE = (255, 80, 80)
        self.COLOR_OBSTACLE_GLOW = (255, 150, 150)
        self.COLOR_FINISH = (80, 255, 80)
        self.COLOR_FINISH_GLOW = (150, 255, 150)
        self.COLOR_UI_TEXT = (240, 240, 255)
        self.COLOR_PARTICLE = (200, 200, 220)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 36)
        
        # --- Internal State ---
        # These are initialized here to satisfy linter, but are properly set in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage_cleared = False
        self._current_stage = 1
        self.timer = 0
        
        self.robot_pos = pygame.Vector2(0, 0)
        self.robot_vel = pygame.Vector2(0, 0)
        self.robot_size = pygame.Vector2(30, 40)
        self.on_ground = True
        self.squash_factor = 1.0

        self.obstacles = []
        self.obstacle_speed = 0.0
        self.finish_line_x = 0
        
        self.particles = []
        
        # Initialize state variables
        # self.reset() is called by the validation function
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Stage Progression Logic ---
        if self.stage_cleared and self._current_stage < self.MAX_STAGES:
            self._current_stage += 1
        else:
            self._current_stage = 1

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage_cleared = False
        self.timer = self.TIME_LIMIT_SECONDS * self.FPS
        
        # Robot state
        self.robot_pos = pygame.Vector2(self.ROBOT_X_POS, self.GROUND_Y - self.robot_size.y)
        self.robot_vel = pygame.Vector2(0, 0)
        self.on_ground = True
        self.squash_factor = 1.0

        # Obstacle state
        self.obstacle_speed = 4.0 + (self._current_stage - 1) * 0.5
        self.obstacles = []
        self.particles = []
        self._generate_obstacles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()
            
        # --- Action Handling ---
        # movement = action[0]  # unused
        space_pressed = action[1] == 1  # Small jump
        shift_pressed = action[2] == 1  # Large jump

        reward = 0.1  # Survival reward

        # --- Update Game Logic ---
        self._handle_input(space_pressed, shift_pressed)
        self._update_robot()
        self._update_obstacles()
        self._update_particles()
        
        # Reward for being off the ground, penalty for being on
        if not self.on_ground:
            reward += 0.1
        else:
            reward -= 0.2

        # --- Collision & Event Checks ---
        # Obstacle clearing reward
        for obs in self.obstacles:
            if not obs['cleared'] and self.robot_pos.x > obs['rect'].right:
                obs['cleared'] = True
                reward += 1.0
                self.score += 10
                
        # Obstacle collision
        robot_rect = pygame.Rect(self.robot_pos.x, self.robot_pos.y, self.robot_size.x, self.robot_size.y * self.squash_factor)
        for obs in self.obstacles:
            if robot_rect.colliderect(obs['rect']):
                reward = -50.0
                self.score -= 500
                self.game_over = True
                # Sound: Player_Hit
                break
        
        # Finish line
        if self.robot_pos.x > self.finish_line_x and not self.game_over:
            self.stage_cleared = True
            self.game_over = True
            time_bonus = max(0, self.timer // self.FPS) * 10
            reward = 100.0 + time_bonus
            self.score += 1000 + time_bonus * 10
            # Sound: Stage_Clear

        # Timer
        self.timer -= 1
        if self.timer <= 0:
            self.game_over = True
            reward = -25.0 # Penalty for running out of time
            # Sound: Time_Out
        
        self.steps += 1
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, space_pressed, shift_pressed):
        if self.on_ground:
            if shift_pressed:
                self.robot_vel.y = self.JUMP_LARGE
                self.on_ground = False
                self.squash_factor = 1.3 # Stretch
                # Sound: Jump_Large
            elif space_pressed:
                self.robot_vel.y = self.JUMP_SMALL
                self.on_ground = False
                self.squash_factor = 1.2 # Stretch
                # Sound: Jump_Small

    def _update_robot(self):
        # Apply gravity
        if not self.on_ground:
            self.robot_vel.y += self.GRAVITY
        
        # Update position
        self.robot_pos.y += self.robot_vel.y
        
        # Squash and stretch decay
        self.squash_factor = max(1.0, self.squash_factor - 0.04)

        # Ground collision
        if self.robot_pos.y + self.robot_size.y >= self.GROUND_Y:
            if not self.on_ground: # Just landed
                self._create_particles(pygame.Vector2(self.robot_pos.x + self.robot_size.x / 2, self.GROUND_Y), 10, self.COLOR_PARTICLE)
                self.squash_factor = 0.7 # Squash
                # Sound: Land
            self.robot_pos.y = self.GROUND_Y - self.robot_size.y
            self.robot_vel.y = 0
            self.on_ground = True

    def _update_obstacles(self):
        # Scroll obstacles and finish line
        for obs in self.obstacles:
            obs['rect'].x -= self.obstacle_speed
        self.finish_line_x -= self.obstacle_speed
        
        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]

    def _generate_obstacles(self):
        current_x = self.WIDTH + 100
        level_length = 4000 + self._current_stage * 1000
        
        while current_x < level_length:
            gap_size = self.np_random.integers(200, 350)
            current_x += gap_size
            
            obstacle_height = self.np_random.integers(40, 120)
            obstacle_width = self.np_random.integers(50, 100)
            
            self.obstacles.append({
                'rect': pygame.Rect(current_x, self.GROUND_Y - obstacle_height, obstacle_width, obstacle_height),
                'cleared': False
            })
            current_x += obstacle_width
        
        self.finish_line_x = level_length + 200

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
            "stage": self._current_stage,
            "time_remaining_steps": self.timer,
            "stage_cleared": self.stage_cleared,
        }

    def _render_game(self):
        self._draw_background()
        self._draw_particles()
        
        # Draw finish line
        if 0 < self.finish_line_x < self.WIDTH:
            self._draw_glowing_line(
                self.screen, self.COLOR_FINISH, self.COLOR_FINISH_GLOW,
                (int(self.finish_line_x), 0), (int(self.finish_line_x), self.HEIGHT), 3
            )

        # Draw obstacles
        for obs in self.obstacles:
            if obs['rect'].right > 0 and obs['rect'].left < self.WIDTH:
                self._draw_glowing_rect(self.screen, self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_GLOW, obs['rect'], 3)

        # Draw ground
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 3)

        # Draw robot
        self._draw_robot()

    def _draw_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _draw_robot(self):
        current_height = self.robot_size.y * self.squash_factor
        current_width = self.robot_size.x / self.squash_factor
        
        center_x = self.robot_pos.x + self.robot_size.x / 2
        bottom_y = self.robot_pos.y + self.robot_size.y
        
        robot_rect = pygame.Rect(
            center_x - current_width / 2,
            bottom_y - current_height,
            current_width,
            current_height
        )
        
        self._draw_glowing_rect(self.screen, self.COLOR_ROBOT, self.COLOR_ROBOT_GLOW, robot_rect, 4, 10)
        
        eye_x = robot_rect.centerx + 5
        eye_y = robot_rect.centery - 5
        pygame.draw.circle(self.screen, (255, 255, 255), (int(eye_x), int(eye_y)), 4)
        pygame.draw.circle(self.screen, (0, 0, 0), (int(eye_x), int(eye_y)), 2)

    def _render_ui(self):
        stage_text = self.font_medium.render(f"Stage: {self._current_stage}/{self.MAX_STAGES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (10, 10))

        time_sec = max(0, self.timer // self.FPS)
        time_ms = max(0, (self.timer % self.FPS) * (1000 // self.FPS))
        timer_text = self.font_medium.render(f"Time: {time_sec:02d}.{time_ms:03d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))
        
        if self.game_over:
            if self.stage_cleared:
                end_text_str = "STAGE CLEAR"
                if self._current_stage == self.MAX_STAGES:
                    end_text_str = "YOU WIN!"
            elif self.timer <= 0:
                end_text_str = "TIME'S UP"
            else:
                end_text_str = "GAME OVER"
                
            end_text = self.font_large.render(end_text_str, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -0.5)),
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += 0.1
            p['lifespan'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]

    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 20))))
            try:
                # Use a surface to handle per-pixel alpha
                radius = int(p['radius'])
                if radius <= 0: continue
                surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, p['color'] + (alpha,), (radius, radius), radius)
                self.screen.blit(surf, (int(p['pos'].x - radius), int(p['pos'].y - radius)))
            except (ValueError, TypeError):
                pass # Ignore errors from invalid alpha/color

    def _draw_glowing_rect(self, surface, color, glow_color, rect, width, glow_radius=5):
        pygame.draw.rect(surface, color, rect, 0, border_radius=3)
        for r in range(glow_radius, 0, -1):
            alpha = int(100 * (1 - r / glow_radius))
            try:
                glow_col = glow_color + (alpha,)
                pygame.draw.rect(surface, glow_col, rect.inflate(r*2, r*2), 1, border_radius=r+3)
            except (ValueError, TypeError):
                pass

    def _draw_glowing_line(self, surface, color, glow_color, start, end, width, glow_radius=5):
        pygame.draw.line(surface, color, start, end, width)
        for r in range(glow_radius, 0, -1):
            alpha = int(100 * (1 - r / glow_radius))
            try:
                glow_col = glow_color + (alpha,)
                pygame.draw.line(surface, glow_col, start, end, width + r)
            except (ValueError, TypeError):
                pass
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    
    pygame.display.init()
    pygame.display.set_caption("Robot Runner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    running = True
    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
        if keys[pygame.K_r]:
            obs, info = env.reset()
            terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()