
# Generated: 2025-08-28T01:28:38.166633
# Source Brief: brief_04116.md
# Brief Index: 4116

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# --- Game Constants ---
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 400
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 15
PADDLE_SPEED = 10
BALL_RADIUS = 8
BALL_SPEED_INITIAL = 6
BLOCK_WIDTH = 58
BLOCK_HEIGHT = 20
BLOCK_ROWS = 10
BLOCK_COLS = 10
UI_MARGIN = 10
MAX_STEPS = 2500

# --- Colors ---
COLOR_BG = (15, 15, 25)
COLOR_PADDLE = (220, 220, 240)
COLOR_BALL = (255, 255, 255)
COLOR_TEXT = (240, 240, 240)
BLOCK_DEFINITIONS = {
    10: (50, 205, 50),    # LimeGreen
    20: (30, 144, 255),   # DodgerBlue
    30: (255, 215, 0),    # Gold
    40: (220, 20, 60),     # Crimson
    50: (148, 0, 211),    # DarkViolet
}

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        self.color = color
        self.np_random = np_random
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = self.np_random.integers(15, 30) # frames
        self.max_lifespan = self.lifespan
        self.radius = self.np_random.uniform(3, 6)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius *= 0.95

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (int(self.x - self.radius), int(self.y - self.radius)), special_flags=pygame.BLEND_RGBA_ADD)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A fast-paced arcade block breaker. Use the paddle to bounce the ball and destroy all the blocks to win."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Initialize state variables to None, to be set in reset()
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = None
        self.blocks = None
        self.total_blocks = 0
        self.blocks_destroyed = 0
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = []
        self.np_random = None
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.paddle_rect = pygame.Rect(
            (SCREEN_WIDTH - PADDLE_WIDTH) / 2,
            SCREEN_HEIGHT - PADDLE_HEIGHT * 2,
            PADDLE_WIDTH,
            PADDLE_HEIGHT,
        )
        
        self._reset_ball()
        
        self.blocks = self._create_blocks()
        self.total_blocks = len(self.blocks)
        self.blocks_destroyed = 0
        
        self.lives = 3
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        blocks = []
        block_colors_list = list(BLOCK_DEFINITIONS.items())
        grid_width = BLOCK_COLS * BLOCK_WIDTH + (BLOCK_COLS - 1) * 5
        start_x = (SCREEN_WIDTH - grid_width) / 2
        start_y = 50

        for i in range(BLOCK_ROWS):
            for j in range(BLOCK_COLS):
                points, color = block_colors_list[i // 2]
                x = start_x + j * (BLOCK_WIDTH + 5)
                y = start_y + i * (BLOCK_HEIGHT + 5)
                block_rect = pygame.Rect(x, y, BLOCK_WIDTH, BLOCK_HEIGHT)
                blocks.append({"rect": block_rect, "points": points, "color": color, "active": True})
        return blocks

    def _reset_ball(self):
        self.ball_attached = True
        self.ball_pos = np.array([self.paddle_rect.centerx, self.paddle_rect.top - BALL_RADIUS], dtype=float)
        self.ball_vel = np.array([0.0, 0.0], dtype=float)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty per step to encourage efficiency

        # 1. Unpack and handle action
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 3:  # Left
            self.paddle_rect.x -= PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += PADDLE_SPEED
        
        self.paddle_rect.clamp_ip(self.screen.get_rect())

        # 2. Update Game Logic
        if self.ball_attached:
            self.ball_pos[0] = self.paddle_rect.centerx
            if space_pressed:
                self.ball_attached = False
                angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
                self.ball_vel = np.array([math.cos(angle), math.sin(angle)]) * BALL_SPEED_INITIAL
                # sfx: launch_ball.wav
        else:
            self.ball_pos += self.ball_vel

        # 3. Handle Collisions and Rewards
        ball_rect = pygame.Rect(self.ball_pos[0] - BALL_RADIUS, self.ball_pos[1] - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos[0] <= BALL_RADIUS or self.ball_pos[0] >= SCREEN_WIDTH - BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], BALL_RADIUS, SCREEN_WIDTH - BALL_RADIUS)
            # sfx: wall_bounce.wav
        if self.ball_pos[1] <= BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], BALL_RADIUS, SCREEN_HEIGHT - BALL_RADIUS)
            # sfx: wall_bounce.wav

        # Paddle collision
        if self.ball_vel[1] > 0 and ball_rect.colliderect(self.paddle_rect):
            self.ball_vel[1] *= -1
            
            offset = (self.ball_pos[0] - self.paddle_rect.centerx) / (PADDLE_WIDTH / 2)
            self.ball_vel[0] = np.clip(self.ball_vel[0] + offset * 2.0, -BALL_SPEED_INITIAL, BALL_SPEED_INITIAL)
            
            norm = np.linalg.norm(self.ball_vel)
            if norm > 0:
                self.ball_vel = self.ball_vel / norm * BALL_SPEED_INITIAL

            # Risk/reward for paddle position
            dist_from_center = abs(self.paddle_rect.centerx - SCREEN_WIDTH / 2)
            normalized_dist = dist_from_center / (SCREEN_WIDTH / 2)
            if normalized_dist > 0.75: reward += 0.5
            elif normalized_dist < 0.25: reward -= 0.2
            # sfx: paddle_bounce.wav

        # Block collisions
        for block in self.blocks:
            if block["active"] and ball_rect.colliderect(block["rect"]):
                block["active"] = False
                self.blocks_destroyed += 1
                
                reward += block["points"] / 10.0
                self.score += block["points"]

                for _ in range(20):
                    self.particles.append(Particle(ball_rect.centerx, ball_rect.centery, block["color"], self.np_random))
                
                # Simple but effective collision response
                self.ball_vel[1] *= -1
                # sfx: block_break.wav
                break

        # Ball out of bounds
        if self.ball_pos[1] > SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 10
            if self.lives > 0:
                self._reset_ball()
                # sfx: lose_life.wav
            else:
                self.game_over = True
                # sfx: game_over.wav

        # 4. Update Particles
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

        # 5. Check Termination Conditions
        win = self.blocks_destroyed == self.total_blocks
        if win:
            reward += 100
            self.game_over = True
            # sfx: win_game.wav

        terminated = self.game_over or self.steps >= MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw blocks
        for block in self.blocks:
            if block["active"]:
                pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
                pygame.draw.rect(self.screen, tuple(min(255, c + 50) for c in block["color"]), block["rect"].inflate(-6, -6), 0, border_radius=2)
                pygame.draw.rect(self.screen, tuple(max(0, c - 50) for c in block["color"]), block["rect"], 2, border_radius=3)

        # Draw paddle
        pygame.draw.rect(self.screen, COLOR_PADDLE, self.paddle_rect, border_radius=3)
        pygame.draw.rect(self.screen, (255,255,255), self.paddle_rect.inflate(-4,-4), 1, border_radius=2)

        # Draw ball with a glow
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], BALL_RADIUS + 3, (*COLOR_BALL, 50))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], BALL_RADIUS + 1, (*COLOR_BALL, 100))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], BALL_RADIUS, COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], BALL_RADIUS, COLOR_BALL)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, COLOR_TEXT)
        self.screen.blit(score_text, (UI_MARGIN, UI_MARGIN))

        # Lives
        life_icon_width = PADDLE_WIDTH / 4
        life_icon_height = PADDLE_HEIGHT / 2
        for i in range(self.lives):
            x = SCREEN_WIDTH - UI_MARGIN - (i + 1) * (life_icon_width + 5)
            y = UI_MARGIN + 5
            pygame.draw.rect(self.screen, COLOR_PADDLE, (x, y, life_icon_width, life_icon_height), border_radius=2)

        # Game Over message
        if self.game_over:
            win = self.blocks_destroyed == self.total_blocks
            message = "YOU WIN!" if win else "GAME OVER"
            color = (100, 255, 100) if win else (255, 100, 100)
            
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_main.render(message, True, color)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_destroyed": self.blocks_destroyed,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    
    # --- Manual Play Setup ---
    obs, info = env.reset()
    terminated = False
    
    display_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Arcade Block Breaker")
    
    running = True
    while running:
        pygame_action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            pygame_action[0] = 3
        elif keys[pygame.K_RIGHT]:
            pygame_action[0] = 4
        
        if keys[pygame.K_SPACE]:
            pygame_action[1] = 1
            
        if keys[pygame.K_r]: # Reset key
            obs, info = env.reset()
            terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(pygame_action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(60)

    env.close()