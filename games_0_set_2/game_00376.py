
# Generated: 2025-08-27T13:28:19.038054
# Source Brief: brief_00376.md
# Brief Index: 376

        
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
    user_guide = "Controls: Press ↑ to jump over obstacles."

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced side-view line runner. Jump over obstacles to reach the 100-meter mark and maximize your score."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_Y = 320
    GOAL_DISTANCE = 100.0
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (15, 23, 42)        # Dark Slate Blue
    COLOR_GRID = (30, 41, 59)      # Lighter Slate
    COLOR_GROUND = (100, 116, 139) # Slate Gray
    COLOR_PLAYER = (255, 255, 255) # White
    COLOR_OBSTACLE = (220, 38, 38) # Red
    COLOR_TEXT = (241, 245, 249)   # Off-white
    COLOR_PARTICLE_RISK = (74, 222, 128) # Green
    COLOR_PARTICLE_CRASH = (249, 115, 22) # Orange

    # Player Physics
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT = 40
    JUMP_STRENGTH = -11
    GRAVITY = 0.5

    # Obstacle Generation
    MIN_OBSTACLE_GAP = 150
    MAX_OBSTACLE_HEIGHT = 60
    MIN_OBSTACLE_HEIGHT = 20
    MAX_OBSTACLE_WIDTH = 30
    MIN_OBSTACLE_WIDTH = 15

    # Difficulty Scaling
    INITIAL_RUNNER_SPEED = 3.0
    INITIAL_OBSTACLE_FREQ = 0.02
    SPEED_INCREMENT = 0.1
    FREQ_INCREMENT = 0.0005
    DIFFICULTY_INTERVAL = 50

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_large = pygame.font.Font(None, 36)
        
        # Initialize state variables to be set in reset()
        self.player_rect = None
        self.player_vy = 0
        self.is_jumping = False
        self.safe_jump_active = False

        self.obstacles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.distance_traveled = 0.0
        self.runner_speed = self.INITIAL_RUNNER_SPEED
        self.obstacle_frequency = self.INITIAL_OBSTACLE_FREQ
        self.last_obstacle_x = 0
        self.game_over = False
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_rect = pygame.Rect(
            self.SCREEN_WIDTH // 4,
            self.GROUND_Y - self.PLAYER_HEIGHT,
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT
        )
        self.player_vy = 0
        self.is_jumping = False
        self.safe_jump_active = False

        self.obstacles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.distance_traveled = 0.0
        self.runner_speed = self.INITIAL_RUNNER_SPEED
        self.obstacle_frequency = self.INITIAL_OBSTACLE_FREQ
        self.last_obstacle_x = self.SCREEN_WIDTH
        self.game_over = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0.1  # Base reward for surviving a step

        # 1. Handle Input
        if movement == 1 and not self.is_jumping:
            self._handle_jump()
            # sfx: jump_sound()

        # 2. Update Game Logic
        self._update_player_physics()
        self._update_world()
        self._update_particles()
        self._generate_obstacles()
        self._scale_difficulty()

        # 3. Collision and Reward Calculation
        collision, terminated, terminal_reward = self._check_collisions_and_goal()
        self.game_over = terminated
        reward += terminal_reward

        if not terminated:
            # Safe jump penalty
            if self.safe_jump_active and self.is_jumping:
                reward -= 0.2
            
            # Risky jump reward
            risky_jump_reward = self._check_risky_jumps()
            reward += risky_jump_reward
            self.score += risky_jump_reward

        self.score += reward
        self.steps += 1

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_jump(self):
        self.is_jumping = True
        self.player_vy = self.JUMP_STRENGTH
        
        # Check if this is a "safe" jump
        is_obstacle_near = False
        for obs in self.obstacles:
            if obs['rect'].left < self.SCREEN_WIDTH: # Check only upcoming obstacles
                is_obstacle_near = True
                break
        self.safe_jump_active = not is_obstacle_near

    def _update_player_physics(self):
        if self.is_jumping:
            self.player_vy += self.GRAVITY
            self.player_rect.y += self.player_vy

        if self.player_rect.bottom >= self.GROUND_Y:
            if self.is_jumping: # Just landed
                # sfx: land_sound()
                self._create_particles(self.player_rect.midbottom, 10, self.COLOR_GROUND, (-2, 2), (-3, -1), 15)
            self.player_rect.bottom = self.GROUND_Y
            self.player_vy = 0
            self.is_jumping = False
            self.safe_jump_active = False

    def _update_world(self):
        # Approximate distance traveled in meters
        meters_per_pixel = self.GOAL_DISTANCE / (self.INITIAL_RUNNER_SPEED * 1200) # Estimate
        self.distance_traveled += self.runner_speed * meters_per_pixel

        for obs in self.obstacles:
            obs['rect'].x -= self.runner_speed
        
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]
        self.last_obstacle_x -= self.runner_speed

    def _generate_obstacles(self):
        if self.last_obstacle_x < self.SCREEN_WIDTH - self.MIN_OBSTACLE_GAP:
            if self.np_random.random() < self.obstacle_frequency:
                width = self.np_random.integers(self.MIN_OBSTACLE_WIDTH, self.MAX_OBSTACLE_WIDTH + 1)
                height = self.np_random.integers(self.MIN_OBSTACLE_HEIGHT, self.MAX_OBSTACLE_HEIGHT + 1)
                new_obstacle = {
                    'rect': pygame.Rect(self.SCREEN_WIDTH, self.GROUND_Y - height, width, height),
                    'cleared': False
                }
                self.obstacles.append(new_obstacle)
                self.last_obstacle_x = self.SCREEN_WIDTH

    def _scale_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.runner_speed += self.SPEED_INCREMENT
            self.obstacle_frequency += self.FREQ_INCREMENT

    def _check_collisions_and_goal(self):
        # Goal reached
        if self.distance_traveled >= self.GOAL_DISTANCE:
            # sfx: victory_sound()
            return False, True, 100

        # Obstacle collision
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs['rect']):
                # sfx: crash_sound()
                self._create_particles(self.player_rect.center, 30, self.COLOR_PARTICLE_CRASH, (-5, 5), (-5, 5), 30)
                return True, True, -100
        
        return False, False, 0
    
    def _check_risky_jumps(self):
        reward = 0
        for obs in self.obstacles:
            is_above = self.player_rect.bottom < obs['rect'].top
            is_horizontally_aligned = self.player_rect.right > obs['rect'].left and self.player_rect.left < obs['rect'].right
            
            if not obs['cleared'] and self.is_jumping and is_above and is_horizontally_aligned:
                obs['cleared'] = True
                reward += 1.0
                # sfx: risky_jump_sound()
                self._create_particles(self.player_rect.midbottom, 15, self.COLOR_PARTICLE_RISK, (-2, 2), (0, 2), 20)
        return reward

    def _create_particles(self, pos, count, color, vx_range, vy_range, lifetime):
        for _ in range(count):
            particle = {
                'pos': list(pos),
                'vel': [self.np_random.uniform(*vx_range), self.np_random.uniform(*vy_range)],
                'color': color,
                'lifetime': self.np_random.integers(lifetime // 2, lifetime + 1),
                'max_lifetime': lifetime
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }
    
    def _render_game(self):
        # Background grid with parallax scrolling
        grid_offset = -(self.distance_traveled * 20) % 40
        for x in range(int(grid_offset), self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.GROUND_Y))
        for y in range(0, self.GROUND_Y, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Ground
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 5)

        # Particles
        for p in self.particles:
            if p['lifetime'] > 0 and p['max_lifetime'] > 0:
                alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
                color = p['color']
                # Using gfxdraw for antialiasing
                pos_x, pos_y = int(p['pos'][0]), int(p['pos'][1])
                pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, 2, (*color, alpha))
                pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, 2, (*color, alpha))

        # Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])

        # Player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)

    def _render_ui(self):
        # Distance
        dist_text = f"DISTANCE: {min(self.distance_traveled, self.GOAL_DISTANCE):.1f} / {self.GOAL_DISTANCE:.0f} m"
        dist_surf = self.font_large.render(dist_text, True, self.COLOR_TEXT)
        self.screen.blit(dist_surf, (10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_surf, score_rect)
        
        # Termination message
        if self.game_over:
            message = ""
            if self.distance_traveled >= self.GOAL_DISTANCE:
                message = "GOAL REACHED!"
            elif self.steps >= self.MAX_STEPS:
                message = "TIME'S UP!"
            else: # Collision
                message = "CRASHED!"

            msg_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            # Add a semi-transparent background for readability
            bg_rect_surf = pygame.Surface(msg_rect.size, pygame.SRCALPHA)
            bg_rect_surf.fill((*self.COLOR_BG, 180))
            self.screen.blit(bg_rect_surf, msg_rect.topleft)
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# This block allows for human play and testing.
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # Create a window for display.
    pygame.display.set_caption("Line Runner")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    action = env.action_space.sample() 
    action.fill(0) # Start with no-op

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get key presses
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Map keys to MultiDiscrete action space
        if keys[pygame.K_UP]:
            action[0] = 1 # up
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(30)

        if done:
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()
            done = False

    env.close()