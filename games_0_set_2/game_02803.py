
# Generated: 2025-08-28T06:00:18.508274
# Source Brief: brief_02803.md
# Brief Index: 2803

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "A fast-paced block-breaking game. Clear all blocks to advance to the next stage. "
        "Maximize your score before the timer runs out or you lose all your balls."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PADDLE = (255, 255, 0)
    COLOR_BALL = (255, 255, 255)
    COLOR_BALL_GLOW = (200, 200, 255)
    COLOR_WALL = (180, 180, 180)
    COLOR_TEXT = (255, 255, 255)
    BLOCK_COLORS = [
        (0, 150, 255),  # 1 HP
        (255, 150, 0),  # 2 HP
        (200, 50, 200),  # 3 HP
    ]

    # Game parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 10
    BALL_RADIUS = 8
    INITIAL_LIVES = 3
    MAX_STAGES = 3
    TIME_PER_STAGE_SECONDS = 60
    FPS = 30 # Game logic is tied to this, not just rendering

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Internal state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_on_paddle = True
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.stage = 1
        self.timer = 0
        self.game_over = False
        self.np_random = None

        # Pre-render background for performance
        self.background = self._create_gradient_background()

        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.stage = 1
        self.game_over = False
        self.particles = []

        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean

        self.steps += 1
        self.timer -= 1

        if not self.game_over:
            # --- Update Game Logic ---
            reward += self._update_paddle(movement)
            reward += self._update_ball(space_held)
            self._update_particles()

            # --- Check Termination Conditions ---
            if self.lives <= 0:
                reward -= 100  # Penalty for losing all lives
                self.game_over = True
                self.game_over_message = "GAME OVER"
            elif self.timer <= 0:
                reward -= 50 # Lesser penalty for timeout
                self.game_over = True
                self.game_over_message = "TIME'S UP"
            elif not self.blocks:
                reward += 5  # Stage clear bonus
                self.stage += 1
                if self.stage > self.MAX_STAGES:
                    reward += 100  # Bonus for winning the game
                    self.game_over = True
                    self.game_over_message = "YOU WIN!"
                else:
                    self._setup_stage() # Progress to next stage

        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _setup_stage(self):
        """Initializes the game state for the current stage."""
        self.timer = self.TIME_PER_STAGE_SECONDS * self.FPS
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_on_paddle = True
        self.blocks.clear()
        
        # Generate blocks
        block_rows = 3 + self.stage
        block_cols = 10
        block_width = self.SCREEN_WIDTH / block_cols
        block_height = 20
        max_hp = min(self.stage, 3)

        for r in range(block_rows):
            for c in range(block_cols):
                hp = self.np_random.integers(1, max_hp + 1)
                block_rect = pygame.Rect(
                    c * block_width,
                    r * block_height + 50,
                    block_width,
                    block_height
                )
                self.blocks.append({"rect": block_rect, "hp": hp})

    def _update_paddle(self, movement):
        """Updates paddle position based on action."""
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = max(0, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH))
        
        # Small penalty for moving to encourage efficiency
        return -0.02 if movement in [3, 4] else 0

    def _update_ball(self, space_held):
        """Updates ball position and handles launch."""
        if self.ball_on_paddle:
            self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
            if space_held:
                # Sound: Ball launch
                self.ball_on_paddle = False
                launch_angle = (self.np_random.random() * 0.4 - 0.2) * math.pi
                base_speed = 6 + (self.stage - 1) * 0.5
                self.ball_vel = [base_speed * math.sin(launch_angle), -base_speed * math.cos(launch_angle)]
                return 0.1 # Small reward for launching
        else:
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
            return self._handle_ball_collisions()
        return 0

    def _handle_ball_collisions(self):
        """Handles all ball collisions and returns rewards."""
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(1, ball_rect.left)
            ball_rect.right = min(self.SCREEN_WIDTH - 1, ball_rect.right)
            self.ball_pos[0] = ball_rect.centerx
            # Sound: Wall bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(1, ball_rect.top)
            self.ball_pos[1] = ball_rect.centery
            # Sound: Wall bounce
            
        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            # Influence horizontal velocity based on hit location
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 4
            # Clamp horizontal speed
            self.ball_vel[0] = max(-10, min(10, self.ball_vel[0]))
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            # Sound: Paddle bounce

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                reward += 0.1 # Reward for hitting a block
                block["hp"] -= 1
                self._create_particles(ball_rect.center, block["hp"])
                if block["hp"] <= 0:
                    reward += 1 # Reward for destroying a block
                    self.score += 10 * self.stage
                    self.blocks.remove(block)
                    # Sound: Block destroy
                else:
                    # Sound: Block hit
                    pass
                
                # Simple bounce logic
                self.ball_vel[1] *= -1
                break # Only handle one block collision per frame

        # Bottom boundary (lose life)
        if ball_rect.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            self.ball_on_paddle = True
            if self.lives > 0:
                # Sound: Lose life
                reward -= 10
            else:
                # Sound: Game over
                pass
        
        # Anti-stuck mechanism
        if not self.ball_on_paddle and abs(self.ball_vel[1]) < 0.5:
            self.ball_vel[1] = 1 if self.ball_vel[1] >= 0 else -1

        return reward

    def _update_particles(self):
        """Updates position and lifetime of particles."""
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
    
    def _create_particles(self, pos, hp):
        """Spawns particles on block hit/destruction."""
        color_index = min(hp, len(self.BLOCK_COLORS) - 1)
        color = self.BLOCK_COLORS[color_index]
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(10, 20),
                "color": color,
                "size": self.np_random.integers(2, 5)
            })

    def _get_observation(self):
        """Renders the game state to an array."""
        self.screen.blit(self.background, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
            "timer": self.timer / self.FPS
        }

    def _render_game(self):
        """Renders all primary game elements."""
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 2)
        
        # Blocks
        for block in self.blocks:
            color = self.BLOCK_COLORS[min(block["hp"] - 1, len(self.BLOCK_COLORS) - 1)]
            inner_rect = block["rect"].inflate(-2, -2)
            pygame.draw.rect(self.screen, color, inner_rect, border_radius=3)
            # Add a subtle highlight
            highlight_color = tuple(min(255, c + 40) for c in color)
            pygame.draw.rect(self.screen, highlight_color, inner_rect.move(0, -2).inflate(0, -inner_rect.height + 4), border_radius=3)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20))
            color = p["color"] + (alpha,)
            s = pygame.Surface((p["size"], p["size"]), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p["pos"][0]), int(p["pos"][1])))

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        # Paddle highlight
        highlight_rect = self.paddle.copy()
        highlight_rect.height = 4
        highlight_rect.y += 2
        pygame.draw.rect(self.screen, (255, 255, 150), highlight_rect, border_radius=3)

        # Ball
        if self.ball_pos:
            x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
            # Glow effect
            glow_radius = int(self.BALL_RADIUS * 1.8)
            glow_alpha = 90
            pygame.gfxdraw.filled_circle(self.screen, x, y, glow_radius, self.COLOR_BALL_GLOW + (glow_alpha,))
            pygame.gfxdraw.aacircle(self.screen, x, y, glow_radius, self.COLOR_BALL_GLOW + (glow_alpha,))
            # Main ball
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        """Renders the UI overlay."""
        # Score
        self._draw_text(f"SCORE: {self.score}", self.font_small, self.COLOR_TEXT, self.screen, 10, 10)
        
        # Timer
        time_str = f"{int(self.timer / self.FPS):02d}"
        self._draw_text(time_str, self.font_large, self.COLOR_TEXT, self.screen, self.SCREEN_WIDTH // 2, 5, align="top")

        # Lives
        life_icon_radius = 8
        for i in range(self.lives):
            x = self.SCREEN_WIDTH - 20 - (i * (life_icon_radius * 2 + 10))
            y = 20
            pygame.gfxdraw.filled_circle(self.screen, x, y, life_icon_radius, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, x, y, life_icon_radius, self.COLOR_PADDLE)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            self._draw_text(self.game_over_message, self.font_large, self.COLOR_TEXT, self.screen, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 30, align="center")

    def _draw_text(self, text, font, color, surface, x, y, align="topleft"):
        """Utility for drawing text with alignment."""
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "topleft":
            text_rect.topleft = (x, y)
        elif align == "top":
            text_rect.midtop = (x, y)
        elif align == "center":
            text_rect.center = (x, y)
        surface.blit(text_surface, text_rect)

    def _create_gradient_background(self):
        """Creates a pre-rendered surface with a vertical gradient."""
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
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

# Example usage to test the environment
if __name__ == '__main__':
    # Set to "human" to visualize, "rgb_array" for headless
    render_mode = "human" 
    
    # Pygame setup for human rendering
    if render_mode == "human":
        pygame.init()
        human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Block Breaker")
        clock = pygame.time.Clock()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    print("Starting game loop...")
    print(f"User Guide: {env.user_guide}")
    print(f"Game Description: {env.game_description}")

    while not done:
        # --- Human Controls ---
        if render_mode == "human":
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

        # --- Agent Controls (random agent) ---
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")

        # --- Render for Human ---
        if render_mode == "human":
            # Convert the observation back to a Pygame surface
            # The observation is (H, W, C), but pygame needs (W, H)
            # and surfarray.make_surface expects (W, H, C)
            # The env returns (H, W, C), and we need to transpose it to (W, H, C) for blitting
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(GameEnv.FPS)

    print("Game Over!")
    print(f"Final Info: {info}")
    env.close()