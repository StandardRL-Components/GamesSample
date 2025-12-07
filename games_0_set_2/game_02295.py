
# Generated: 2025-08-28T04:21:45.023481
# Source Brief: brief_02295.md
# Brief Index: 2295

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ for a small jump, ↓ for a large jump. Survive and reach the end!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a robot through a procedurally generated obstacle course. Time your jumps to perfection to reach the finish line. Difficulty increases through 3 stages."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_ROBOT = (60, 160, 255)
        self.COLOR_OBSTACLE = (255, 80, 80)
        self.COLOR_OBSTACLE_HL = (255, 150, 150)
        self.COLOR_FINISH = (80, 255, 80)
        self.COLOR_PARTICLE = (200, 200, 220)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_ORANGE = (255, 165, 0)
        self.COLOR_GRAY = (128, 128, 128)
        self.COLOR_TEXT = (230, 230, 230)

        # Physics and Player
        self.GRAVITY = 0.8
        self.JUMP_SMALL = -12
        self.JUMP_LARGE = -16
        self.ROBOT_SPEED = 5
        self.GROUND_Y = 350
        self.ROBOT_SCREEN_X = 100
        self.ROBOT_WIDTH = 24
        self.ROBOT_HEIGHT = 36

        # Game progression
        self.STAGE_LENGTH = 6000 # pixels
        self.FINISH_LINE_X = self.STAGE_LENGTH * 3
        self.MAX_TOTAL_STEPS = 30 * 180 # 180 seconds total

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 60, bold=True)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.stage = 1
        self.stage_timer = 0
        self.world_offset_x = 0
        self.robot_pos = [0, 0]
        self.robot_vel_y = 0
        self.is_grounded = True
        self.obstacles = []
        self.next_obstacle_x = 0
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.stage = 1
        self.stage_timer = 60 * self.FPS

        self.world_offset_x = 0
        self.robot_pos = [self.ROBOT_SCREEN_X, self.GROUND_Y]
        self.robot_vel_y = 0
        self.is_grounded = True
        
        self.obstacles = []
        self.next_obstacle_x = 400
        self._generate_obstacles()

        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if not self.game_over:
            movement = action[0]

            # 1. Handle Input
            if self.is_grounded:
                if movement == 1: # Up for small jump
                    self.robot_vel_y = self.JUMP_SMALL
                    self.is_grounded = False
                    # sfx: small_jump
                elif movement == 2: # Down for large jump
                    self.robot_vel_y = self.JUMP_LARGE
                    self.is_grounded = False
                    # sfx: large_jump

            # 2. Update Physics
            self.robot_vel_y += self.GRAVITY
            self.robot_pos[1] += self.robot_vel_y
            self.world_offset_x += self.ROBOT_SPEED
            reward += 0.01 # Small reward for surviving/moving forward

            # 3. Ground and Collision Check
            if self.robot_pos[1] >= self.GROUND_Y:
                if not self.is_grounded: # Just landed
                    self._create_landing_particles(20)
                    # sfx: land
                self.robot_pos[1] = self.GROUND_Y
                self.robot_vel_y = 0
                self.is_grounded = True

            robot_rect = pygame.Rect(
                self.robot_pos[0], self.robot_pos[1] - self.ROBOT_HEIGHT, 
                self.ROBOT_WIDTH, self.ROBOT_HEIGHT
            )
            for obs in self.obstacles:
                obs_rect = pygame.Rect(
                    obs['x'] - self.world_offset_x, obs['y'], obs['w'], obs['h']
                )
                if robot_rect.colliderect(obs_rect):
                    self.game_over = True
                    reward = -5
                    # sfx: collision_hit
                    break
            
            # 4. Update Game State
            self.steps += 1
            self.stage_timer -= 1
            self._update_particles()
            self._generate_obstacles()
            self._prune_elements()

            # 5. Stage Progression
            current_stage_end = self.stage * self.STAGE_LENGTH
            if self.world_offset_x > current_stage_end and self.stage < 3:
                self.stage += 1
                reward += 10
                self.stage_timer = 60 * self.FPS
                # sfx: stage_complete
            
            # 6. Termination Conditions
            if self.world_offset_x >= self.FINISH_LINE_X:
                self.game_over = True
                self.victory = True
                reward = 100
                # sfx: victory
            
            if self.stage_timer <= 0 or self.steps >= self.MAX_TOTAL_STEPS:
                self.game_over = True
                if not self.victory:
                    reward = -10 # Timeout penalty

        terminated = self.game_over
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_obstacles()
        self._render_finish_line()
        self._render_particles()
        self._render_robot()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_left": self.stage_timer / self.FPS
        }

    def _render_background(self):
        # Parallax grid
        parallax_factor = 0.2
        grid_size = 50
        offset_x = -(self.world_offset_x * parallax_factor) % grid_size
        for i in range(self.SCREEN_WIDTH // grid_size + 2):
            x = int(i * grid_size + offset_x)
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for i in range(self.SCREEN_HEIGHT // grid_size + 1):
            y = int(i * grid_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GRID, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

    def _render_robot(self):
        robot_rect = pygame.Rect(
            int(self.robot_pos[0]), 
            int(self.robot_pos[1] - self.ROBOT_HEIGHT), 
            self.ROBOT_WIDTH, 
            self.ROBOT_HEIGHT
        )
        # Body with outline
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, robot_rect, 1, border_radius=4)
        
        # Eye
        eye_y = robot_rect.y + 10
        pygame.draw.circle(self.screen, self.COLOR_WHITE, (robot_rect.centerx + 4, eye_y), 4)
        
        # Jetpack flame for jump feedback
        if not self.is_grounded and self.robot_vel_y < 1:
            flame_color = self.COLOR_ORANGE if self.robot_vel_y < 0 else self.COLOR_GRAY
            flame_height = int(max(0, min(15, -self.robot_vel_y * 1.5)))
            flame_width = self.ROBOT_WIDTH - 10 + self.np_random.integers(-2, 3)
            flame_rect = pygame.Rect(
                robot_rect.centerx - flame_width // 2, 
                robot_rect.bottom, 
                flame_width,
                flame_height
            )
            pygame.draw.ellipse(self.screen, flame_color, flame_rect)

    def _render_obstacles(self):
        for obs in self.obstacles:
            x = int(obs['x'] - self.world_offset_x)
            if x < self.SCREEN_WIDTH and x + obs['w'] > 0:
                rect = pygame.Rect(x, int(obs['y']), int(obs['w']), int(obs['h']))
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_HL, (rect.x, rect.y, rect.w, 4)) # Highlight

    def _render_finish_line(self):
        finish_x = int(self.FINISH_LINE_X - self.world_offset_x)
        if finish_x < self.SCREEN_WIDTH:
            check_size = 20
            for i in range(self.SCREEN_HEIGHT // check_size):
                color = self.COLOR_FINISH if i % 2 == 0 else self.COLOR_WHITE
                pygame.draw.rect(self.screen, color, (finish_x, i * check_size, check_size, check_size))

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p['life'] / 6))
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.circle(self.screen, p['color'], pos, size)

    def _render_ui(self):
        # Stage text
        stage_text = self.font_ui.render(f"STAGE: {self.stage}/3", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Timer text
        time_left = max(0, self.stage_timer / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))
        
        # Score text
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))

        # Game Over / Victory text
        if self.game_over:
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_FINISH if self.victory else self.COLOR_OBSTACLE
            end_text = self.font_big.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _generate_obstacles(self):
        gap_ranges = {1: (200, 400), 2: (150, 300), 3: (100, 220)}
        min_gap, max_gap = gap_ranges[self.stage]

        while self.next_obstacle_x < self.world_offset_x + self.SCREEN_WIDTH + 200:
            gap = self.np_random.integers(min_gap, max_gap + 1)
            width = self.np_random.integers(30, 80 + 1)
            height = self.np_random.integers(40, 120 + 1)
            
            x_pos = self.next_obstacle_x + gap
            y_pos = self.GROUND_Y - height

            self.obstacles.append({'x': x_pos, 'y': y_pos, 'w': width, 'h': height})
            self.next_obstacle_x = x_pos + width

    def _prune_elements(self):
        self.obstacles = [obs for obs in self.obstacles if obs['x'] + obs['w'] > self.world_offset_x]
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_landing_particles(self, count):
        for _ in range(count):
            self.particles.append({
                'pos': [self.robot_pos[0] + self.ROBOT_WIDTH / 2, self.GROUND_Y],
                'vel': [self.np_random.uniform(-2.5, 2.5), self.np_random.uniform(-3, -1)],
                'life': self.np_random.integers(15, 30),
                'color': self.COLOR_PARTICLE
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # particle gravity
            p['life'] -= 1

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Robot Runner")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Default action is no-op
        action = [0, 0, 0] 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1 # Small jump
        elif keys[pygame.K_DOWN]:
            action[0] = 2 # Large jump

        if keys[pygame.K_r]: # Press R to reset
             obs, info = env.reset()
             total_reward = 0
             done = False
             continue

        if done:
            # In a done state, we can only reset
            continue

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()