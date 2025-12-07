
# Generated: 2025-08-27T16:12:01.723349
# Source Brief: brief_01152.md
# Brief Index: 1152

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Ensure Pygame runs headlessly
os.environ['SDL_VIDEODRIVER'] = 'dummy'

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Arrow keys to move. Hold Space for a speed boost."
    )
    game_description = (
        "Guide a robot through a procedural obstacle course. Move close to "
        "obstacles for bonus points, but don't crash!"
    )

    # Frame advance behavior
    auto_advance = False

    # Class-level attributes for persistent state across episodes
    total_steps_across_episodes = 0
    BASE_OBSTACLES = 5
    DIFFICULTY_INTERVAL = 500
    DIFFICULTY_INCREASE = 0.05

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PLAYER = (60, 160, 255)
        self.COLOR_PLAYER_ACCENT = (200, 220, 255)
        self.COLOR_OBSTACLE = (255, 80, 100)
        self.COLOR_FINISH = (80, 255, 120)
        self.COLOR_PARTICLE = (255, 220, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_SUCCESS = (120, 255, 150)
        self.COLOR_FAILURE = (255, 120, 120)

        # Game element properties
        self.PLAYER_SIZE = 16
        self.FINISH_LINE_WIDTH = 20
        self.RISKY_MOVE_DISTANCE = 40
        self.MIN_OBSTACLE_SIZE = 20
        self.MAX_OBSTACLE_SIZE = 60

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
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Initialize State ---
        self.player_rect = None
        self.obstacles = []
        self.finish_rect = None
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_outcome = "" # "SUCCESS" or "FAILURE"
        self.last_dist_to_goal = 0.0
        
        # This will be properly initialized in reset()
        self.np_random = None

        # --- Final Validation ---
        # self.validate_implementation() # Commented out for submission, but useful for dev

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_outcome = ""
        self.particles.clear()

        # Place player
        player_y = self.HEIGHT // 2 - self.PLAYER_SIZE // 2
        self.player_rect = pygame.Rect(30, player_y, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Place finish line
        self.finish_rect = pygame.Rect(
            self.WIDTH - self.FINISH_LINE_WIDTH - 10, 0, self.FINISH_LINE_WIDTH, self.HEIGHT
        )
        
        # Procedurally generate obstacles
        self._generate_obstacles()

        self.last_dist_to_goal = self._get_distance_to_goal()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is already over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Processing ---
        movement, space_held, _ = action
        is_boost = space_held == 1
        move_speed = 2 if is_boost else 1
        
        dx, dy = 0, 0
        if movement == 1: dy = -move_speed  # Up
        elif movement == 2: dy = move_speed   # Down
        elif movement == 3: dx = -move_speed  # Left
        elif movement == 4: dx = move_speed   # Right

        # --- Game Logic ---
        self.steps += 1
        GameEnv.total_steps_across_episodes += 1
        
        prev_player_pos = self.player_rect.center
        self.player_rect.move_ip(dx, dy)
        self._clamp_player_to_bounds()

        # --- Reward Calculation ---
        reward = -0.01  # Small penalty for each step to encourage speed

        # Reward for moving closer to the goal
        current_dist_to_goal = self._get_distance_to_goal()
        reward += self.last_dist_to_goal - current_dist_to_goal
        self.last_dist_to_goal = current_dist_to_goal

        # Reward/penalty for risk-taking
        dist_to_obstacle = self._get_distance_to_nearest_obstacle()
        if dist_to_obstacle < self.RISKY_MOVE_DISTANCE and (dx != 0 or dy != 0):
            reward += 5.0
            # SFX: Risky move whoosh
            self._spawn_particles(prev_player_pos, 5, self.COLOR_PARTICLE)
        else:
            reward -= 0.5
        
        if is_boost and (dx != 0 or dy != 0):
            # SFX: Boost sound
            self._spawn_particles(prev_player_pos, 2, self.COLOR_PLAYER_ACCENT, life=5)


        # --- Collision & Termination Check ---
        terminated = False
        if self.player_rect.colliderect(self.finish_rect):
            # SFX: Victory fanfare
            reward += 100.0
            self.game_over = True
            terminated = True
            self.game_outcome = "SUCCESS"
        elif any(self.player_rect.colliderect(obs) for obs in self.obstacles):
            # SFX: Explosion sound
            reward -= 50.0
            self.game_over = True
            terminated = True
            self.game_outcome = "FAILURE"
            self._spawn_particles(self.player_rect.center, 30, self.COLOR_OBSTACLE, life=40)
        elif self.steps >= self.MAX_STEPS:
            # SFX: Timeout buzzer
            terminated = True
            self.game_over = True
            self.game_outcome = "TIMEOUT"

        self.score += reward

        # --- Update Particles ---
        self._update_particles()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_observation(self):
        # Clear screen with background color
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Draw background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw finish line
        pygame.draw.rect(self.screen, self.COLOR_FINISH, self.finish_rect)

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p['color'], alpha), (size, size), size)
                self.screen.blit(s, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))

        # Draw player if not crashed
        if self.game_outcome != "FAILURE":
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)
            inner_rect = self.player_rect.inflate(-self.PLAYER_SIZE // 2, -self.PLAYER_SIZE // 2)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_ACCENT, inner_rect)

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"SCORE: {self.score:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer/Steps display
        steps_text = self.font_main.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        # Game over message
        if self.game_over:
            if self.game_outcome == "SUCCESS":
                msg_text = self.font_big.render("SUCCESS", True, self.COLOR_SUCCESS)
            elif self.game_outcome == "FAILURE":
                msg_text = self.font_big.render("FAILURE", True, self.COLOR_FAILURE)
            else: # TIMEOUT
                msg_text = self.font_big.render("TIMEOUT", True, self.COLOR_TEXT)

            text_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_text, text_rect)

    def _generate_obstacles(self):
        self.obstacles.clear()
        
        # Calculate number of obstacles based on persistent difficulty
        difficulty_multiplier = 1 + (GameEnv.total_steps_across_episodes // self.DIFFICULTY_INTERVAL) * self.DIFFICULTY_INCREASE
        num_obstacles = int(self.BASE_OBSTACLES * difficulty_multiplier)

        spawn_area = pygame.Rect(100, 20, self.WIDTH - 200, self.HEIGHT - 40)

        for _ in range(num_obstacles):
            placed = False
            while not placed:
                w = self.np_random.integers(self.MIN_OBSTACLE_SIZE, self.MAX_OBSTACLE_SIZE + 1)
                h = self.np_random.integers(self.MIN_OBSTACLE_SIZE, self.MAX_OBSTACLE_SIZE + 1)
                x = self.np_random.integers(spawn_area.left, spawn_area.right - w)
                y = self.np_random.integers(spawn_area.top, spawn_area.bottom - h)
                new_obs = pygame.Rect(x, y, w, h)

                # Ensure it doesn't block the start or end immediately
                if not new_obs.colliderect(self.player_rect.inflate(40,40)) and not new_obs.colliderect(self.finish_rect.inflate(40,40)):
                    self.obstacles.append(new_obs)
                    placed = True
    
    def _clamp_player_to_bounds(self):
        self.player_rect.left = max(0, self.player_rect.left)
        self.player_rect.right = min(self.WIDTH, self.player_rect.right)
        self.player_rect.top = max(0, self.player_rect.top)
        self.player_rect.bottom = min(self.HEIGHT, self.player_rect.bottom)

    def _get_distance_to_goal(self):
        return math.hypot(
            self.player_rect.centerx - self.finish_rect.centerx,
            self.player_rect.centery - self.finish_rect.centery
        )

    def _get_distance_to_nearest_obstacle(self):
        if not self.obstacles:
            return float('inf')
        
        min_dist = float('inf')
        player_center = self.player_rect.center
        for obs in self.obstacles:
            # Find the closest point on the obstacle to the player
            closest_x = max(obs.left, min(player_center[0], obs.right))
            closest_y = max(obs.top, min(player_center[1], obs.bottom))
            dist = math.hypot(player_center[0] - closest_x, player_center[1] - closest_y)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _spawn_particles(self, pos, count, color, life=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            # Add some friction/drag
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # This part is for demonstration and will not be part of the final submission
    # It requires a display, so we unset the dummy driver
    if 'SDL_VIDEODRIVER' in os.environ:
        del os.environ['SDL_VIDEODRIVER']

    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    obs, info = env.reset(seed=42)
    
    # Setup a visible pygame window
    pygame.display.set_caption("Robot Obstacle Course")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    running = True
    
    # Game loop
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        else:
            # Allow reset on key press after game over
            if any(keys):
                obs, info = env.reset()
                terminated = False

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for interactive play

    env.close()