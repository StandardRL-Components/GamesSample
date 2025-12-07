
# Generated: 2025-08-27T14:33:17.839691
# Source Brief: brief_00718.md
# Brief Index: 718

        
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
        "A retro arcade block breaker. Clear all blocks across 3 stages. Risky plays are rewarded."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Game settings
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STAGES = 3
    STAGE_TIME_SECONDS = 60
    
    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_PADDLE_GLOW = (200, 200, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 255, 150)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (50, 50, 80)
    BLOCK_COLORS = [
        (255, 50, 50), (50, 255, 50), (50, 150, 255),
        (255, 128, 0), (150, 50, 255), (0, 255, 255)
    ]
    BLOCK_HIT_COLOR = (200, 200, 200)

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
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        
        # Internal state variables are initialized in reset()
        self.paddle = None
        self.ball = None
        self.blocks = []
        self.particles = []
        self.score = 0
        self.balls_left = 0
        self.stage = 0
        self.stage_timer = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.score = 0
        self.balls_left = 3
        self.stage = 1
        self.game_over = False
        self.game_won = False
        self.steps = 0
        
        self.paddle = pygame.Rect(
            self.SCREEN_WIDTH / 2 - 50, self.SCREEN_HEIGHT - 30, 100, 15
        )
        self.paddle_speed = 12

        self.particles = []
        self._setup_stage(self.stage)
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self, stage_num):
        self.blocks = []
        self.stage_timer = self.STAGE_TIME_SECONDS * self.FPS
        
        # Reset ball state
        ball_base_speed = 4.5
        ball_speed_increase = 1.0 + (stage_num - 1) * 0.25 # 1.0, 1.25, 1.5
        ball_speed = ball_base_speed * ball_speed_increase

        self.ball = {
            "pos": pygame.Vector2(self.paddle.centerx, self.paddle.top - 10),
            "vel": pygame.Vector2(0, 0),
            "radius": 7,
            "launched": False,
            "speed": ball_speed
        }
        
        # Block layouts
        block_width, block_height = 50, 20
        rows, cols = 0, 0
        layout = []

        if stage_num == 1:
            rows, cols = 4, 10
            layout = [[1]*cols for _ in range(rows)]
        elif stage_num == 2:
            rows, cols = 6, 10
            layout = [[0]*cols for _ in range(rows)]
            for r in range(rows):
                for c in range(cols):
                    if (r + c) % 2 == 0:
                        layout[r][c] = 1
        elif stage_num == 3:
            rows, cols = 7, 10
            layout = [[0]*cols for _ in range(rows)]
            for r in range(rows):
                layout[r][0] = layout[r][-1] = 2
            for c in range(cols):
                layout[0][c] = 2
            layout[3][3:7] = [2,2,2,2]

        y_offset = 50
        for r in range(rows):
            for c in range(cols):
                if layout[r][c] > 0:
                    block_rect = pygame.Rect(
                        c * (block_width + 4) + 38,
                        r * (block_height + 4) + y_offset,
                        block_width,
                        block_height
                    )
                    color_index = (r + c) % len(self.BLOCK_COLORS)
                    self.blocks.append({
                        "rect": block_rect,
                        "color": self.BLOCK_COLORS[color_index],
                        "hits": layout[r][c]
                    })
    
    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean, we care about the press event
        # shift_held is unused per brief

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # -- Update game logic --
        self.steps += 1
        self.stage_timer -= 1

        # 1. Paddle Movement
        if movement == 3: # Left
            self.paddle.x -= self.paddle_speed
        elif movement == 4: # Right
            self.paddle.x += self.paddle_speed
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.paddle.width)

        # 2. Ball Launch
        if space_pressed and not self.ball["launched"]:
            self.ball["launched"] = True
            # Sound: Ball Launch
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            self.ball["vel"] = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.ball["speed"]
        
        # 3. Ball Movement & Collisions
        if self.ball["launched"]:
            self.ball["pos"] += self.ball["vel"]
            reward += self._handle_collisions()
        else:
            self.ball["pos"].x = self.paddle.centerx
            reward -= 0.02 # Penalty for not playing

        # 4. Particle updates
        self._update_particles()

        # 5. Check for stage clear
        if not self.blocks:
            reward += 5.0
            self.stage += 1
            if self.stage > self.MAX_STAGES:
                self.game_won = True
            else:
                # Sound: Stage Clear
                self._setup_stage(self.stage)

        # 6. Check for termination conditions
        terminated = self._check_termination()
        if terminated:
            if self.game_won:
                reward += 100.0
            else:
                reward -= 100.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball["pos"].x - self.ball["radius"], self.ball["pos"].y - self.ball["radius"], self.ball["radius"] * 2, self.ball["radius"] * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball["vel"].x *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.SCREEN_WIDTH, ball_rect.right)
            # Sound: Wall Bounce
        if ball_rect.top <= 0:
            self.ball["vel"].y *= -1
            ball_rect.top = max(0, ball_rect.top)
            # Sound: Wall Bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball["vel"].y > 0:
            # Sound: Paddle Hit
            self.ball["vel"].y *= -1
            
            # Add horizontal influence
            offset = (self.paddle.centerx - self.ball["pos"].x) / (self.paddle.width / 2)
            self.ball["vel"].x -= offset * 2.0
            
            # Normalize to maintain constant speed
            if self.ball["vel"].length() > 0:
                 self.ball["vel"].scale_to_length(self.ball["speed"])

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                # Sound: Block Hit
                reward += 0.1
                block["hits"] -= 1
                
                if block["hits"] <= 0:
                    reward += 1.0
                    self._create_particles(block["rect"].center, block["color"])
                    self.blocks.remove(block)
                    # Sound: Block Break
                    self.score += 10
                
                # Bounce logic
                self.ball["vel"].y *= -1 # Simple vertical bounce
                break # Handle one collision per frame

        # Ball lost
        if ball_rect.top > self.SCREEN_HEIGHT:
            self.balls_left -= 1
            reward -= 0.2
            self.ball["launched"] = False
            self.ball["pos"] = pygame.Vector2(self.paddle.centerx, self.paddle.top - 10)
            self.ball["vel"] = pygame.Vector2(0, 0)
            # Sound: Lose Life
        
        return reward

    def _check_termination(self):
        if self.game_won:
            self.game_over = True
            return True
        if self.balls_left <= 0 or self.stage_timer <= 0:
            self.game_over = True
            return True
        return False

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "radius": self.np_random.uniform(1, 4),
                "color": color,
                "life": self.np_random.integers(15, 30)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Drag
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, (25, 25, 45), (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, (25, 25, 45), (0, i), (self.SCREEN_WIDTH, i))

        # Draw blocks
        for block in self.blocks:
            color = block["color"] if block["hits"] == 1 else self.BLOCK_HIT_COLOR
            pygame.draw.rect(self.screen, tuple(c*0.6 for c in color), block["rect"])
            pygame.draw.rect(self.screen, color, block["rect"].inflate(-4, -4))

        # Draw paddle with glow
        glow_rect = self.paddle.inflate(8, 8)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PADDLE_GLOW + (30,), glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with glow
        pos_int = (int(self.ball["pos"].x), int(self.ball["pos"].y))
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(self.ball["radius"] * 1.8), self.COLOR_BALL_GLOW + (80,))
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ball["radius"], self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ball["radius"], self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color)

    def _render_text(self, text, font, pos, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW):
        text_surf = font.render(text, True, shadow_color)
        self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        score_text = f"SCORE: {self.score:05d}"
        balls_text = f"BALLS: {self.balls_left}"
        stage_text = f"STAGE: {self.stage}"
        time_left = max(0, self.stage_timer // self.FPS)
        timer_text = f"TIME: {time_left:02d}"

        self._render_text(score_text, self.font_small, (10, 10))
        self._render_text(balls_text, self.font_small, (220, 10))
        self._render_text(stage_text, self.font_small, (380, 10))
        self._render_text(timer_text, self.font_small, (500, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            text_surf = self.font_large.render(msg, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2 + 3, self.SCREEN_HEIGHT/2 + 3))
            self.screen.blit(text_surf, text_rect)
            
            text_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "balls_left": self.balls_left,
            "timer": self.stage_timer // self.FPS,
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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup for manual play
    pygame.display.set_caption("Arcade Block Breaker")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        clock.tick(env.FPS)
        
    env.close()