
# Generated: 2025-08-28T06:45:49.196248
# Source Brief: brief_03029.md
# Brief Index: 3029

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-futuristic Block Breaker. Clear all the blocks to win, but lose all your balls and you lose."
    )

    # Frames auto-advance at a fixed rate for smooth, real-time gameplay.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2500

    # Colors (Neon/Retro-Futuristic)
    COLOR_BG = (10, 10, 26) # Dark blue-purple
    COLOR_GRID = (30, 30, 60)
    COLOR_PADDLE = (0, 255, 255) # Cyan
    COLOR_BALL = (255, 255, 0) # Yellow
    COLOR_TEXT = (220, 220, 255)
    
    BLOCK_COLORS = {
        1: (0, 255, 128),   # Green
        3: (0, 128, 255),   # Blue
        5: (128, 0, 255),   # Purple
        10: (255, 192, 0),  # Gold
    }

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
        
        # Fonts
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 0
        self.ball_on_paddle = True
        self.rng = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.ball_on_paddle = True

        # Paddle setup
        paddle_width, paddle_height = 100, 15
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - paddle_width) / 2, 
            self.SCREEN_HEIGHT - paddle_height - 10, 
            paddle_width, paddle_height
        )
        
        # Ball setup
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - 8)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_radius = 7

        # Procedural block generation
        self._generate_blocks()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.02 # Small penalty per step to encourage speed
        self.game_over = self._check_termination()
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_held = action[1] == 1  # Boolean
            
            # --- Handle Input ---
            if movement == 3:  # Left
                self.paddle.x -= 12
            elif movement == 4: # Right
                self.paddle.x += 12
            
            # Keep paddle within screen bounds
            self.paddle.x = max(0, min(self.SCREEN_WIDTH - self.paddle.width, self.paddle.x))

            # Launch ball
            if self.ball_on_paddle and space_held:
                # Sound: Ball launch
                self.ball_on_paddle = False
                initial_angle = self.rng.uniform(-math.pi * 0.75, -math.pi * 0.25)
                self.ball_vel = pygame.Vector2(math.cos(initial_angle), math.sin(initial_angle)) * 6
                reward += 0.1 # Small reward for launching

            # --- Update Game State ---
            if self.ball_on_paddle:
                self.ball_pos.x = self.paddle.centerx
            else:
                reward += self._update_ball()
            
            self._update_particles()

        # Check for termination conditions and apply terminal rewards
        self.game_over = self._check_termination()
        if self.game_over:
            if not self.blocks: # Win condition
                reward += 100
            elif self.balls_left <= 0: # Lose condition
                reward += -100

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _generate_blocks(self):
        self.blocks = []
        block_width, block_height = 40, 20
        rows, cols = 5, 12
        
        for r in range(rows):
            for c in range(cols):
                # Leave gaps for solvability
                if self.rng.random() < 0.85:
                    x = c * (block_width + 5) + 35
                    y = r * (block_height + 5) + 40
                    
                    points = 1
                    if r < 1: points = 10 # Gold
                    elif r < 2: points = 5 # Purple
                    elif r < 3: points = 3 # Blue

                    block_rect = pygame.Rect(x, y, block_width, block_height)
                    self.blocks.append({
                        "rect": block_rect,
                        "points": points,
                        "color": self.BLOCK_COLORS[points]
                    })

    def _update_ball(self):
        reward = 0
        self.ball_pos += self.ball_vel

        # Wall collisions
        if self.ball_pos.x - self.ball_radius <= 0 or self.ball_pos.x + self.ball_radius >= self.SCREEN_WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.ball_radius, min(self.SCREEN_WIDTH - self.ball_radius, self.ball_pos.x))
            # Sound: Wall bounce
        if self.ball_pos.y - self.ball_radius <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.ball_radius
            # Sound: Wall bounce

        # Bottom wall (lose ball)
        if self.ball_pos.y + self.ball_radius >= self.SCREEN_HEIGHT:
            self.balls_left -= 1
            reward -= 5
            self.ball_on_paddle = True
            self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - 8)
            self.ball_vel = pygame.Vector2(0, 0)
            # Sound: Lose ball

        # Paddle collision
        ball_rect = pygame.Rect(self.ball_pos.x - self.ball_radius, self.ball_pos.y - self.ball_radius, self.ball_radius * 2, self.ball_radius * 2)
        if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
            reward += 0.1
            # Sound: Paddle hit
            
            # Calculate bounce angle based on hit location
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.paddle.width / 2)
            bounce_angle = math.radians(90 - offset * 75) # Max 75 degree angle change
            
            # Maintain ball speed
            speed = self.ball_vel.length()
            self.ball_vel.x = math.cos(bounce_angle) * speed
            self.ball_vel.y = -abs(math.sin(bounce_angle) * speed)
            
            # Ensure ball speed doesn't get too low
            if self.ball_vel.length() < 5:
                self.ball_vel = self.ball_vel.normalize() * 5

        # Block collisions
        for block in self.blocks[:]:
            if block["rect"].colliderect(ball_rect):
                # Sound: Block break
                reward += block["points"]
                self.score += block["points"]
                self._create_particles(block["rect"].center, block["color"])
                self.blocks.remove(block)
                
                # Determine collision side to correctly reverse velocity
                prev_ball_pos = self.ball_pos - self.ball_vel
                if (prev_ball_pos.x < block["rect"].left or prev_ball_pos.x > block["rect"].right):
                    self.ball_vel.x *= -1
                else:
                    self.ball_vel.y *= -1
                break
        
        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "life": self.rng.integers(20, 40),
                "color": color,
                "radius": self.rng.uniform(1, 3)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Damping
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.balls_left <= 0 or not self.blocks

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

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

    def _render_glow_rect(self, surf, rect, color, glow_size=10, glow_alpha=70):
        glow_rect = rect.inflate(glow_size, glow_size)
        
        # Create a temporary surface for the glow
        temp_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        
        # Draw a filled rectangle on it
        pygame.draw.rect(temp_surf, (*color, glow_alpha), (0, 0, *glow_rect.size), border_radius=int(glow_size/2))
        
        # Scale it up and then down to create a blur effect
        scaled_size = (glow_rect.width * 2, glow_rect.height * 2)
        temp_surf = pygame.transform.smoothscale(temp_surf, scaled_size)
        temp_surf = pygame.transform.smoothscale(temp_surf, glow_rect.size)
        
        surf.blit(temp_surf, glow_rect.topleft)

    def _render_glow_circle(self, surf, pos, radius, color, glow_size=15, glow_alpha=90):
        # Draw multiple transparent circles to simulate a glow
        for i in range(glow_size, 0, -2):
            alpha = glow_alpha * (1 - i / glow_size)
            pygame.gfxdraw.filled_circle(
                surf, int(pos[0]), int(pos[1]),
                int(radius + i),
                (*color, int(alpha))
            )

    def _render_game(self):
        # Background grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Render blocks
        for block in self.blocks:
            self._render_glow_rect(self.screen, block["rect"], block["color"], glow_size=8, glow_alpha=60)
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in block["color"]), block["rect"].inflate(-6, -6), border_radius=2)

        # Render paddle
        self._render_glow_rect(self.screen, self.paddle, self.COLOR_PADDLE, glow_size=12, glow_alpha=80)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Render ball
        self._render_glow_circle(self.screen, self.ball_pos, self.ball_radius, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.ball_radius, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.ball_radius, self.COLOR_BALL)

        # Render particles
        for p in self.particles:
            alpha = 255 * (p["life"] / 40)
            color = (*p["color"], alpha)
            pygame.draw.circle(self.screen, color, p["pos"], p["radius"])

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Balls left
        ball_icon_radius = 8
        for i in range(self.balls_left):
            pos_x = self.SCREEN_WIDTH - 20 - (i * (ball_icon_radius * 2 + 10))
            pos_y = 15
            self._render_glow_circle(self.screen, (pos_x, pos_y), ball_icon_radius, self.COLOR_PADDLE, glow_size=8, glow_alpha=60)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, ball_icon_radius, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, ball_icon_radius, self.COLOR_PADDLE)
            
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "LEVEL CLEARED" if not self.blocks else "GAME OVER"
            text_surf = self.font_main.render(win_text, True, self.COLOR_BALL if not self.blocks else (255, 50, 50))
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    total_reward = 0
    
    # Set up the display window
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    
    while running:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Not used in this game
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        # --- Rendering ---
        # The observation is already the rendered screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to reset.")
            
        # Control the frame rate
        env.clock.tick(60)

    env.close()