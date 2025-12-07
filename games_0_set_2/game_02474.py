import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A top-down block breaker where strategic ball bouncing and risk-taking are rewarded."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 8
    BALL_SPEED = 5
    BLOCK_ROWS, BLOCK_COLS = 5, 10
    BLOCK_WIDTH, BLOCK_HEIGHT = 60, 20
    BLOCK_SPACING = 4
    UI_HEIGHT = 40
    TOTAL_LIVES = 3
    TOTAL_STAGES = 3
    STAGE_TIME_SECONDS = 60

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 255, 0, 50)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (255, 87, 34), (255, 193, 7), (139, 195, 74),
        (0, 188, 212), (33, 150, 243), (156, 39, 176)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        self.np_random = None

        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = None
        self.blocks = []
        self.particles = []
        self.score = 0
        self.lives = 0
        self.stage = 0
        self.stage_timer = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.lives = self.TOTAL_LIVES
        self.stage = 1
        self.steps = 0
        self.game_over = False
        self.game_won = False

        self._setup_stage()

        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.particles = []
        self.stage_timer = self.STAGE_TIME_SECONDS * 60  # 60 FPS
        self._reset_paddle_and_ball()
        self._generate_blocks()

    def _reset_paddle_and_ball(self):
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_attached = True

    def _generate_blocks(self):
        self.blocks = []
        total_block_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.WIDTH - total_block_width) / 2
        start_y = self.UI_HEIGHT + 20

        for r in range(self.BLOCK_ROWS):
            for c in range(self.BLOCK_COLS):
                x = start_x + c * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = start_y + r * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                
                # Simple stage layouts
                if self.stage == 1 and (r % 2 == 0 or c % 2 == 0):
                    continue
                if self.stage == 2 and (c < 2 or c > self.BLOCK_COLS - 3):
                    continue
                if self.stage == 3 and (r + c) % 2 != 0:
                    continue

                color = self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({"rect": block_rect, "color": color, "active": True})

        if sum(1 for b in self.blocks if b["active"]) == 0: # Ensure stage isn't empty
            for i in range(50):
                r = self.np_random.integers(0, self.BLOCK_ROWS)
                c = self.np_random.integers(0, self.BLOCK_COLS)
                x = start_x + c * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = start_y + r * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                color = self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({"rect": block_rect, "color": color, "active": True})


    def step(self, action):
        if self.auto_advance:
            self.clock.tick(60)

        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1

        reward = -0.01  # Time penalty
        self.steps += 1
        self.stage_timer -= 1

        # --- Player Actions ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        if movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        if self.ball_attached and space_held:
            # sfx: ball_launch
            self.ball_attached = False
            # FIX: The `high` argument must be greater than the `low` argument.
            # The original code had -math.pi/3 (approx -1.047) as low and
            # -2*math.pi/3 (approx -2.094) as high, which caused the error.
            angle = self.np_random.uniform(-2 * math.pi / 3, -math.pi / 3)
            self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED

        # --- Ball Physics ---
        if not self.ball_attached:
            reward += 0.1  # Reward for keeping ball in play
            self.ball_pos += self.ball_vel

            ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # Wall collisions
            if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
                self.ball_vel.x *= -1
                self.ball_pos.x = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos.x))
                # sfx: ball_bounce_wall
            if self.ball_pos.y - self.BALL_RADIUS <= self.UI_HEIGHT:
                self.ball_vel.y *= -1
                self.ball_pos.y = self.UI_HEIGHT + self.BALL_RADIUS
                # sfx: ball_bounce_wall

            # Paddle collision
            if ball_rect.colliderect(self.paddle):
                # sfx: ball_bounce_paddle
                self.ball_vel.y *= -1
                self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
                
                # Influence horizontal velocity based on hit location
                offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel.x += offset * 2
                self.ball_vel.normalize_ip()
                self.ball_vel *= self.BALL_SPEED

            # Block collisions
            for block in self.blocks:
                if block["active"] and ball_rect.colliderect(block["rect"]):
                    # sfx: block_break
                    block["active"] = False
                    reward += 1.0
                    self.score += 10
                    self._spawn_particles(block["rect"].center, block["color"])

                    # Collision response
                    prev_ball_pos = self.ball_pos - self.ball_vel
                    prev_ball_rect = pygame.Rect(prev_ball_pos.x - self.BALL_RADIUS, prev_ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
                    
                    if prev_ball_rect.bottom <= block["rect"].top or prev_ball_rect.top >= block["rect"].bottom:
                        self.ball_vel.y *= -1
                    else:
                        self.ball_vel.x *= -1
                    break 

            # Ball lost
            if self.ball_pos.y > self.HEIGHT:
                # sfx: life_lost
                self.lives -= 1
                reward -= 5.0
                if self.lives > 0:
                    self._reset_paddle_and_ball()
                else:
                    self.game_over = True

        # --- Stage Progression ---
        if not any(b["active"] for b in self.blocks):
            # sfx: stage_clear
            reward += 10.0
            self.score += 100
            self.stage += 1
            if self.stage > self.TOTAL_STAGES:
                self.game_won = True
                reward += 100.0
            else:
                self._setup_stage()
        
        # --- Time Up ---
        if self.stage_timer <= 0:
            self.game_over = True
            reward -= 5.0 # Penalty for running out of time

        # --- Particle Update ---
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

        terminated = self.game_over or self.game_won
        truncated = False
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "life": self.np_random.integers(20, 40),
                "color": color,
                "radius": self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame surface arrays are (width, height, channels).
        # Gymnasium expects (height, width, channels).
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 40))
            color_with_alpha = p["color"] + (alpha,)
            # Use a simple circle for particles to avoid surface creation overhead
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color_with_alpha)

        # Draw blocks
        for block in self.blocks:
            if block["active"]:
                darker_color = tuple(max(0, c - 40) for c in block["color"])
                pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
                pygame.draw.rect(self.screen, darker_color, block["rect"], width=2, border_radius=3)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Draw ball
        if self.ball_attached:
            self.ball_pos.x = self.paddle.centerx
        
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), glow_radius, self.COLOR_BALL_GLOW)
        # Ball itself
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # UI Background
        pygame.draw.rect(self.screen, (0,0,0,100), (0, 0, self.WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_TEXT, (0, self.UI_HEIGHT), (self.WIDTH, self.UI_HEIGHT), 1)

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.stage_timer // 60)
        timer_color = self.COLOR_TEXT if time_left > 10 else (255, 100, 100)
        timer_text = self.font_small.render(f"TIME: {time_left}", True, timer_color)
        self.screen.blit(timer_text, (150, 10))

        # Lives
        lives_text = self.font_small.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 150, 10))
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 80 + i * 20, 22, 6, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 80 + i * 20, 22, 6, self.COLOR_BALL)

        # Stage
        stage_text = self.font_large.render(f"- STAGE {self.stage} -", True, self.COLOR_TEXT)
        text_rect = stage_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 25))
        self.screen.blit(stage_text, text_rect)

        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER"
        elif self.game_won:
            msg = "YOU WIN!"
        else:
            msg = None
        
        if msg:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(msg, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Create a display window for human play
    pygame.display.init()
    pygame.display.set_caption("Block Breaker")
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        action = np.array([movement, space_held, 0]) # Shift is unused

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame
        # Pygame uses (width, height), numpy uses (height, width)
        # We need to transpose it back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()