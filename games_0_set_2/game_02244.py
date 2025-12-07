
# Generated: 2025-08-27T19:44:41.057198
# Source Brief: brief_02244.md
# Brief Index: 2244

        
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
    user_guide = "Controls: ←→ to move the paddle."

    # Must be a short, user-facing description of the game:
    game_description = (
        "A grid-based Breakout variant. Break bricks, collect power-ups, "
        "and reach the target score before you run out of balls."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 80, 10
    PADDLE_SPEED = 10
    BALL_RADIUS = 6
    INITIAL_BALL_SPEED = 4.0
    BRICK_ROWS, BRICK_COLS = 8, 14
    BRICK_WIDTH, BRICK_HEIGHT = 40, 15
    BRICK_GAP = 4
    UI_HEIGHT = 40
    MAX_STEPS = 5000
    INITIAL_BALLS = 5

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_PADDLE = (220, 220, 230)
    COLOR_BALL = (255, 255, 0)
    COLOR_GRID = (30, 35, 50)
    # Brick types
    BRICK_EMPTY = 0
    BRICK_STANDARD = 1
    BRICK_BONUS = 2
    BRICK_PENALTY = 3
    BRICK_POWERUP = 4
    BRICK_COLORS = {
        BRICK_STANDARD: (60, 120, 220),
        BRICK_BONUS: (60, 220, 120),
        BRICK_PENALTY: (220, 60, 60),
        BRICK_POWERUP: (255, 180, 0),
    }
    # Powerup types
    POWERUP_EXTEND = 1
    POWERUP_DOUBLE = 2
    POWERUP_SLOW = 3
    POWERUP_COLORS = {
        POWERUP_EXTEND: (180, 180, 255),
        POWERUP_DOUBLE: (255, 255, 100),
        POWERUP_SLOW: (100, 200, 255),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Initialize state variables
        self.game_over = False
        self.steps = 0
        self.score = 0
        self.stage = 0
        self.balls_left = 0
        self.paddle = None
        self.balls = []
        self.bricks = []
        self.particles = []
        self.falling_powerups = []
        self.active_powerups = {}
        self.stage_target_scores = {1: 150, 2: 300, 3: 500}
        self.np_random = None
        
        # This will be initialized in reset()
        self.reset()
        
        # Self-validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.balls_left = self.INITIAL_BALLS
        self.particles = []
        self.falling_powerups = []
        self.active_powerups = {}

        self._setup_stage(self.stage)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for time passing
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self._move_paddle(movement)
        reward += self._update_balls()
        reward += self._update_powerups()
        self._update_particles()
        
        self.steps += 1
        
        # Check for stage clear
        if self.stage < 3 and self.score >= self.stage_target_scores[self.stage]:
            self.stage += 1
            self._setup_stage(self.stage)
            reward += 5.0 # Stage clear reward

        # Check for termination conditions
        terminated = False
        if self.score >= self.stage_target_scores[3]: # Win
            self.game_over = True
            terminated = True
            reward += 100.0
        elif self.balls_left <= 0 and not self.balls: # Loss
            self.game_over = True
            terminated = True
            reward += -100.0
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _setup_stage(self, stage_num):
        # Reset paddle
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - 30,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.active_powerups.clear()
        self.falling_powerups.clear()

        # Reset ball(s)
        self.balls = [self._create_ball()]

        # Create brick layout
        self.bricks = []
        grid_width = self.BRICK_COLS * (self.BRICK_WIDTH + self.BRICK_GAP) - self.BRICK_GAP
        start_x = (self.WIDTH - grid_width) // 2
        start_y = self.UI_HEIGHT + 20
        
        brick_density = 0.5 + (stage_num - 1) * 0.1
        
        for r in range(self.BRICK_ROWS):
            for c in range(self.BRICK_COLS):
                if self.np_random.random() < brick_density:
                    brick_type = self.np_random.choice(
                        [self.BRICK_STANDARD, self.BRICK_BONUS, self.BRICK_PENALTY, self.BRICK_POWERUP],
                        p=[0.7, 0.1, 0.1, 0.1]
                    )
                    x = start_x + c * (self.BRICK_WIDTH + self.BRICK_GAP)
                    y = start_y + r * (self.BRICK_HEIGHT + self.BRICK_GAP)
                    self.bricks.append({
                        "rect": pygame.Rect(x, y, self.BRICK_WIDTH, self.BRICK_HEIGHT),
                        "type": brick_type
                    })

    def _create_ball(self, pos=None, vel=None):
        if pos is None:
            pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1)
        if vel is None:
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            speed = self.INITIAL_BALL_SPEED + (self.stage - 1) * 0.2
            if self.active_powerups.get(self.POWERUP_SLOW, 0) > 0:
                speed *= 0.6
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        
        return {"pos": pos, "vel": vel, "radius": self.BALL_RADIUS}

    def _move_paddle(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.WIDTH, self.paddle.right)

    def _update_balls(self):
        step_reward = 0
        
        for ball in self.balls[:]:
            ball["pos"] += ball["vel"]
            ball_rect = pygame.Rect(ball["pos"].x - ball["radius"], ball["pos"].y - ball["radius"], ball["radius"] * 2, ball["radius"] * 2)

            # Wall collisions
            if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
                ball["vel"].x *= -1
                ball["pos"].x = max(ball["radius"], min(self.WIDTH - ball["radius"], ball["pos"].x))
                # sfx: bounce_wall
            if ball_rect.top <= self.UI_HEIGHT:
                ball["vel"].y *= -1
                ball["pos"].y = max(self.UI_HEIGHT + ball["radius"], ball["pos"].y)
                # sfx: bounce_wall

            # Paddle collision
            if ball_rect.colliderect(self.paddle) and ball["vel"].y > 0:
                # sfx: bounce_paddle
                offset = (ball_rect.centerx - self.paddle.centerx) / (self.paddle.width / 2)
                offset = max(-1, min(1, offset))
                
                speed = ball["vel"].length()
                ball["vel"].x = offset * speed * 0.9 # 0.9 to give more verticality
                ball["vel"].y = -math.sqrt(max(0.01, speed**2 - ball["vel"].x**2))
                ball["pos"].y = self.paddle.top - ball["radius"]

            # Brick collisions
            hit_brick = False
            for brick in self.bricks[:]:
                if ball_rect.colliderect(brick["rect"]):
                    hit_brick = True
                    # sfx: hit_brick
                    self.bricks.remove(brick)
                    
                    # Collision response
                    overlap = ball_rect.clip(brick["rect"])
                    if overlap.width < overlap.height:
                        ball["vel"].x *= -1
                    else:
                        ball["vel"].y *= -1
                    
                    # Score and rewards
                    if brick["type"] == self.BRICK_STANDARD:
                        self.score += 1
                        step_reward += 0.1
                    elif brick["type"] == self.BRICK_BONUS:
                        self.score += 5
                        step_reward += 0.5
                    elif brick["type"] == self.BRICK_PENALTY:
                        self.score -= 2
                        step_reward -= 0.2
                    elif brick["type"] == self.BRICK_POWERUP:
                        self.score += 1
                        self._spawn_powerup(brick["rect"].center)

                    # Create particles
                    for _ in range(10):
                        self.particles.append(self._create_particle(brick["rect"].center, self.BRICK_COLORS[brick["type"]]))

                    break # Only handle one brick collision per frame per ball
            
            # Ball lost
            if ball_rect.top >= self.HEIGHT:
                self.balls.remove(ball)
                # sfx: lose_ball
                if not self.balls: # If it was the last ball
                    self.balls_left -= 1
                    if self.balls_left > 0:
                        self.balls.append(self._create_ball())
        
        return step_reward

    def _spawn_powerup(self, pos):
        powerup_type = self.np_random.choice([self.POWERUP_EXTEND, self.POWERUP_DOUBLE, self.POWERUP_SLOW])
        self.falling_powerups.append({
            "pos": pygame.Vector2(pos),
            "type": powerup_type,
            "size": 10
        })

    def _update_powerups(self):
        reward = 0
        # Update falling powerups
        for p in self.falling_powerups[:]:
            p["pos"].y += 2
            p_rect = pygame.Rect(p["pos"].x - p["size"], p["pos"].y - p["size"], p["size"]*2, p["size"]*2)
            if p_rect.colliderect(self.paddle):
                # sfx: powerup_collect
                self._activate_powerup(p["type"])
                self.falling_powerups.remove(p)
                reward += 1.0
            elif p_rect.top > self.HEIGHT:
                self.falling_powerups.remove(p)

        # Update active powerups
        keys_to_del = []
        for p_type, duration in self.active_powerups.items():
            if duration > 0:
                self.active_powerups[p_type] -= 1
            if self.active_powerups[p_type] <= 0:
                keys_to_del.append(p_type)
        
        for key in keys_to_del:
            self._deactivate_powerup(key)
            del self.active_powerups[key]
        
        return reward

    def _activate_powerup(self, p_type):
        if p_type == self.POWERUP_EXTEND:
            if not self.active_powerups.get(p_type, 0) > 0:
                self.paddle.width *= 1.5
                self.paddle.inflate_ip(self.paddle.width * 0.5, 0)
            self.active_powerups[p_type] = 300 # 10 seconds at 30fps
        
        elif p_type == self.POWERUP_DOUBLE:
            if len(self.balls) < 5: # Max 5 balls on screen
                self.balls.append(self._create_ball(pos=pygame.Vector2(self.paddle.center)))
        
        elif p_type == self.POWERUP_SLOW:
            if not self.active_powerups.get(p_type, 0) > 0:
                for ball in self.balls:
                    ball["vel"] *= 0.6
            self.active_powerups[p_type] = 300 # 10 seconds

    def _deactivate_powerup(self, p_type):
        if p_type == self.POWERUP_EXTEND:
            self.paddle.width = self.PADDLE_WIDTH
        elif p_type == self.POWERUP_SLOW:
            for ball in self.balls:
                ball["vel"] *= (1/0.6)

    def _create_particle(self, pos, color):
        return {
            "pos": pygame.Vector2(pos),
            "vel": pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
            "radius": self.np_random.uniform(2, 5),
            "lifetime": self.np_random.integers(15, 30),
            "color": color,
        }

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Bricks
        for brick in self.bricks:
            color = self.BRICK_COLORS[brick["type"]]
            pygame.draw.rect(self.screen, color, brick["rect"])
            pygame.draw.rect(self.screen, tuple(min(255, c+30) for c in color), brick["rect"], 2)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / 30))
            color = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color
            )

        # Falling powerups
        for p in self.falling_powerups:
            color = self.POWERUP_COLORS[p["type"]]
            rect = pygame.Rect(p["pos"].x - p["size"], p["pos"].y - p["size"], p["size"]*2, p["size"]*2)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, (255,255,255), rect, 2, border_radius=5)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Balls
        for ball in self.balls:
            # Glow effect
            for i in range(4, 0, -1):
                alpha = 80 - i * 20
                pygame.gfxdraw.filled_circle(
                    self.screen, int(ball["pos"].x), int(ball["pos"].y),
                    ball["radius"] + i, (255, 255, 150, alpha)
                )
            pygame.gfxdraw.filled_circle(
                self.screen, int(ball["pos"].x), int(ball["pos"].y), ball["radius"], self.COLOR_BALL
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(ball["pos"].x), int(ball["pos"].y), ball["radius"], self.COLOR_BALL
            )

    def _render_ui(self):
        # UI background bar
        pygame.draw.rect(self.screen, self.COLOR_GRID, (0, 0, self.WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_PADDLE, (0, self.UI_HEIGHT), (self.WIDTH, self.UI_HEIGHT), 1)

        # Score text
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_PADDLE)
        self.screen.blit(score_text, (10, 10))
        
        # Stage text
        stage_text = self.font_small.render(f"STAGE: {self.stage}", True, self.COLOR_PADDLE)
        self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width() // 2, 10))

        # Balls left
        balls_text = self.font_small.render(f"BALLS: {self.balls_left}", True, self.COLOR_PADDLE)
        self.screen.blit(balls_text, (self.WIDTH - balls_text.get_width() - 10, 10))

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.stage_target_scores[3]:
                msg = "YOU WIN!"
                color = self.COLOR_BALL
            else:
                msg = "GAME OVER"
                color = self.BRICK_COLORS[self.BRICK_PENALTY]

            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "balls_left": self.balls_left,
        }

    def close(self):
        pygame.quit()
        
    def render(self):
        # This method is not strictly required by the brief but is useful for human play.
        # It returns the same data as _get_observation but is part of the public API.
        return self._get_observation()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Breakout Gym Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # space/shift not used

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()