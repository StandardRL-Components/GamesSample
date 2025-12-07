
# Generated: 2025-08-27T15:15:46.582666
# Source Brief: brief_00936.md
# Brief Index: 936

        
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
    """
    A Gymnasium environment for a classic Snake game with a minimalist, neon aesthetic.

    The agent controls a snake on a grid, aiming to eat orbs to grow longer and
    increase its score. The episode ends if the snake collides with the walls,
    itself, reaches the maximum score, or exceeds the step limit.

    **Visuals:**
    - Dark, grid-based background.
    - Bright green snake that grows with each orb consumed.
    - A pulsating yellow orb that respawns randomly when eaten.
    - Particle effects upon orb consumption for visual feedback.

    **Rewards:**
    - A large positive reward for eating an orb or winning the game.
    - A large negative penalty for collisions (self or wall).
    - Small shaping rewards for moving towards the orb.
    - Small shaping penalties for moving towards a wall or for each step taken.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Use arrow keys to change the snake's direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a growing snake to devour glowing orbs and achieve a high score "
        "before colliding with yourself or the walls."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_GRID = (30, 30, 40)
        self.COLOR_SNAKE_HEAD = (100, 255, 100)
        self.COLOR_SNAKE_BODY = (0, 200, 80)
        self.COLOR_ORB = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_GAMEOVER = (255, 50, 50)
        
        # Fonts
        self.font_score = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_gameover = pygame.font.SysFont("impact", 60)
        
        # Game state variables
        self.snake_body = []
        self.direction = (0, 0)
        self.orb_pos = (0, 0)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.dist_to_orb = 0
        self.particles = []
        self.orb_pulse_timer = 0
        
        # Action mapping
        self.action_map = {
            1: (0, -1),  # Up
            2: (0, 1),   # Down
            3: (-1, 0),  # Left
            4: (1, 0),   # Right
        }
        
        # Initialize state variables
        self.reset()
        
        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.orb_pulse_timer = 0
        self.particles.clear()
        
        # Initialize snake
        start_x = self.GRID_WIDTH // 2
        start_y = self.GRID_HEIGHT // 2
        self.snake_body = [
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y),
        ]
        self.direction = (1, 0)  # Move right initially
        
        # Place orb
        self._place_orb()
        
        # Calculate initial distance to orb for reward
        head = self.snake_body[0]
        self.dist_to_orb = self._manhattan_distance(head, self.orb_pos)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        
        # Store old state for reward calculation
        old_head = self.snake_body[0]
        old_dist_to_orb = self.dist_to_orb
        old_dist_to_wall = self._dist_to_wall(old_head)

        # 1. Update direction based on action
        if movement in self.action_map:
            new_direction = self.action_map[movement]
            # Prevent reversing on itself
            if len(self.snake_body) > 1 and (new_direction[0] == -self.direction[0] and new_direction[1] == -self.direction[1]):
                pass # Ignore reverse commands
            else:
                self.direction = new_direction
        
        # 2. Update snake position
        new_head = (old_head[0] + self.direction[0], old_head[1] + self.direction[1])
        self.snake_body.insert(0, new_head)
        
        # 3. Handle game logic and calculate rewards
        reward = 0
        terminated = False
        ate_orb = False

        # Check for orb consumption
        if new_head == self.orb_pos:
            self.score += 10
            reward = 10  # Event-based reward
            ate_orb = True
            # sound effect: # eat_sound.play()
            self._create_particles(self.orb_pos)
            self._place_orb()
        else:
            self.snake_body.pop()

        # 4. Check for termination conditions
        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= self.GRID_HEIGHT):
            self.game_over = True
            terminated = True
            reward = -50 # Terminal penalty
            # sound effect: # crash_sound.play()
        # Self collision
        elif new_head in self.snake_body[1:]:
            self.game_over = True
            terminated = True
            reward = -50 # Terminal penalty
            # sound effect: # crash_sound.play()
            
        # Check for win condition
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            terminated = True
            reward = 100 # Terminal reward
            # sound effect: # win_sound.play()

        # Update steps and check for max steps termination
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            self.game_over = True
            terminated = True

        # 5. Continuous/shaping rewards (only if not a terminal/event step)
        if not ate_orb and not terminated:
            reward = -0.01 # Penalty for time passing

            # Reward for moving closer to the orb
            self.dist_to_orb = self._manhattan_distance(new_head, self.orb_pos)
            if self.dist_to_orb < old_dist_to_orb:
                reward += 2
            
            # Penalty for moving closer to a wall
            new_dist_to_wall = self._dist_to_wall(new_head)
            if new_dist_to_wall < old_dist_to_wall:
                reward -= 5

        # 6. Update animations
        self._update_particles()
        self.orb_pulse_timer += 0.2
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
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
            "snake_length": len(self.snake_body),
            "orb_pos": self.orb_pos,
        }

    def _render_game(self):
        self._render_grid()
        self._render_orb()
        self._render_snake()
        self._render_particles()

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_snake(self):
        # Draw body
        for i, segment in enumerate(reversed(self.snake_body[1:])):
            # Fade color along the tail
            fade_factor = (i / max(1, len(self.snake_body) - 1)) * 0.7 + 0.3
            color = (
                int(self.COLOR_SNAKE_BODY[0] * fade_factor),
                int(self.COLOR_SNAKE_BODY[1] * fade_factor),
                int(self.COLOR_SNAKE_BODY[2] * fade_factor)
            )
            rect = pygame.Rect(
                segment[0] * self.GRID_SIZE,
                segment[1] * self.GRID_SIZE,
                self.GRID_SIZE,
                self.GRID_SIZE
            )
            pygame.draw.rect(self.screen, color, rect.inflate(-4, -4), border_radius=4)

        # Draw head
        head = self.snake_body[0]
        head_rect = pygame.Rect(
            head[0] * self.GRID_SIZE,
            head[1] * self.GRID_SIZE,
            self.GRID_SIZE,
            self.GRID_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, head_rect.inflate(-2, -2), border_radius=5)

    def _render_orb(self):
        pulse = (math.sin(self.orb_pulse_timer) + 1) / 2  # Varies between 0 and 1
        radius = int(self.GRID_SIZE * 0.3 + pulse * self.GRID_SIZE * 0.15)
        pixel_pos = (
            int((self.orb_pos[0] + 0.5) * self.GRID_SIZE),
            int((self.orb_pos[1] + 0.5) * self.GRID_SIZE)
        )
        
        # Draw glow
        glow_radius = int(radius * (1.5 + pulse * 0.5))
        glow_alpha = int(100 + pulse * 50)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_ORB, glow_alpha), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (pixel_pos[0] - glow_radius, pixel_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw core orb using anti-aliased drawing
        pygame.gfxdraw.aacircle(self.screen, pixel_pos[0], pixel_pos[1], radius, self.COLOR_ORB)
        pygame.gfxdraw.filled_circle(self.screen, pixel_pos[0], pixel_pos[1], radius, self.COLOR_ORB)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(life_ratio * 4)
            if radius > 0:
                color = (
                    int(self.COLOR_ORB[0] * life_ratio),
                    int(self.COLOR_ORB[1] * life_ratio),
                    int(self.COLOR_ORB[2] * life_ratio)
                )
                pygame.draw.circle(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), radius)

    def _render_ui(self):
        score_text = self.font_score.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            text = "YOU WON!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = self.COLOR_SNAKE_HEAD if self.score >= self.WIN_SCORE else self.COLOR_GAMEOVER
            game_over_surf = self.font_gameover.render(text, True, color)
            pos = (
                self.SCREEN_WIDTH / 2 - game_over_surf.get_width() / 2,
                self.SCREEN_HEIGHT / 2 - game_over_surf.get_height() / 2
            )
            self.screen.blit(game_over_surf, pos)

    def _place_orb(self):
        while True:
            pos = (
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )
            if pos not in self.snake_body:
                self.orb_pos = pos
                break

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _dist_to_wall(self, pos):
        dist_x = min(pos[0], self.GRID_WIDTH - 1 - pos[0])
        dist_y = min(pos[1], self.GRID_HEIGHT - 1 - pos[1])
        return dist_x + dist_y

    def _create_particles(self, grid_pos):
        pixel_x = (grid_pos[0] + 0.5) * self.GRID_SIZE
        pixel_y = (grid_pos[1] + 0.5) * self.GRID_SIZE
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(20, 40)
            self.particles.append(
                {'pos': [pixel_x, pixel_y], 'vel': velocity, 'life': lifetime, 'max_life': lifetime}
            )

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95 # friction
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
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

if __name__ == "__main__":
    # To run and visualize the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame window for visualization
    pygame.display.set_caption("Snake Gym Environment")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0] # Default action is no-op
        
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
        
        if keys[pygame.K_r]: # Reset key
            terminated = False
            obs, info = env.reset()

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(15) # Control game speed for human play

    pygame.quit()