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

    # Short, user-facing control string
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball. "
        "Press shift to activate a collected power-up."
    )

    # Short, user-facing description of the game
    game_description = (
        "A fast-paced, grid-based block breaker. Clear all blocks using the "
        "paddle and ball, and collect powerful upgrades."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    # Screen
    WIDTH, HEIGHT = 640, 400
    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (255, 0, 128), (0, 255, 255), (0, 255, 0), (255, 128, 0), (128, 0, 255)
    ]
    RAINBOW_COLORS = [
        (255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0),
        (0, 0, 255), (75, 0, 130), (148, 0, 211)
    ]
    # Game parameters
    PADDLE_WIDTH, PADDLE_HEIGHT = 80, 12
    PADDLE_SPEED = 10
    BALL_RADIUS = 7
    BALL_SPEED = 6
    MAX_LIVES = 3
    MAX_STEPS = 5000 # Increased for better playability
    POWERUP_CHANCE = 0.25
    POWERUP_SIZE = 15
    POWERUP_SPEED = 2
    POWERUP_DURATION_STEPS = 600  # 20 seconds at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 40, bold=True)

        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.lives = None
        self.game_over = None
        self.paddle = None
        self.balls = None
        self.blocks = None
        self.particles = None
        self.falling_powerups = None
        self.held_powerup = None
        self.powerup_timers = None
        self.is_paddle_extended = None
        self.is_paddle_sticky = None
        self.last_space_held = None
        self.last_shift_held = None
        self.rainbow_idx = 0

        # The reset call is needed to initialize the state for the first time.
        # It's important that reset() is self-contained and doesn't rely on
        # a specific state from __init__ beyond the initial None values.
        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False

        # Paddle
        paddle_y = self.HEIGHT - 30
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        # Effects and Powerups (FIX: Initialized before _reset_ball is called)
        self.particles = []
        self.falling_powerups = []
        self.held_powerup = None
        self.powerup_timers = {"extended": 0, "sticky": 0}
        self.is_paddle_extended = False
        self.is_paddle_sticky = False

        # Ball
        self.balls = []
        self._reset_ball()

        # Blocks
        self.blocks = []
        block_width, block_height = 58, 18
        gap = 6
        for r in range(5):  # 5 rows
            for c in range(10):  # 10 columns
                block_x = c * (block_width + gap) + gap
                block_y = r * (block_height + gap) + 40
                self.blocks.append({
                    "rect": pygame.Rect(block_x, block_y, block_width, block_height),
                    "color": self.BLOCK_COLORS[r]
                })

        # Input state
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage action
        self.steps += 1
        self.rainbow_idx = (self.rainbow_idx + 1) % len(self.RAINBOW_COLORS)

        # --- Action Handling ---
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        self._handle_input(movement, space_pressed, shift_pressed)
        
        # --- Game Logic ---
        step_reward = self._update_game_state()
        reward += step_reward
        
        # --- Termination ---
        terminated = False
        truncated = False
        if self.lives <= 0:
            terminated = True
            reward = -100
        elif not self.blocks:
            terminated = True
            reward = 100
        
        if self.steps >= self.MAX_STEPS:
            truncated = True

        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Paddle movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.paddle.width)

        # Launch attached balls
        if space_pressed:
            for ball in self.balls:
                if ball["attached"]:
                    ball["attached"] = False
                    ball["vel"] = pygame.Vector2(0, -self.BALL_SPEED)
                    # sfx: launch_ball
                elif self.is_paddle_sticky:
                    # Relaunch stuck ball
                    ball["attached"] = False
                    self._bounce_ball(ball, self.paddle)


        # Activate power-up
        if shift_pressed and self.held_powerup is not None:
            self._activate_powerup()
            reward = 10 # This reward is currently not being used in the step function
            self.score += 500

    def _update_game_state(self):
        reward = 0
        self._update_powerup_timers()
        
        reward += self._update_balls()
        reward += self._update_falling_powerups()
        
        self._update_particles()
        return reward

    def _update_balls(self):
        reward = 0
        balls_to_remove = []
        for ball in self.balls:
            if ball["attached"]:
                ball["pos"].x = self.paddle.centerx
                ball["pos"].y = self.paddle.top - self.BALL_RADIUS
                continue

            ball["pos"] += ball["vel"]

            # Wall collisions
            if ball["pos"].x <= self.BALL_RADIUS or ball["pos"].x >= self.WIDTH - self.BALL_RADIUS:
                ball["vel"].x *= -1
                ball["pos"].x = np.clip(ball["pos"].x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
                # sfx: bounce_wall
            if ball["pos"].y <= self.BALL_RADIUS:
                ball["vel"].y *= -1
                ball["pos"].y = self.BALL_RADIUS
                # sfx: bounce_wall

            # Bottom boundary (lose life)
            if ball["pos"].y >= self.HEIGHT:
                balls_to_remove.append(ball)
                continue

            # Paddle collision
            ball_rect = pygame.Rect(ball["pos"].x - self.BALL_RADIUS, ball["pos"].y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            if self.paddle.colliderect(ball_rect):
                if self.is_paddle_sticky:
                    ball["attached"] = True
                    # sfx: sticky_hit
                else:
                    self._bounce_ball(ball, self.paddle)
                    # sfx: bounce_paddle
                ball["pos"].y = self.paddle.top - self.BALL_RADIUS # Prevent getting stuck
                
            # Block collisions
            for block in self.blocks[:]:
                if block["rect"].colliderect(ball_rect):
                    self.blocks.remove(block)
                    ball["vel"].y *= -1 # Simple bounce
                    self.score += 100
                    reward += 0.1
                    self._create_particles(block["rect"].center, block["color"])
                    # sfx: break_block
                    if self.np_random.random() < self.POWERUP_CHANCE:
                        self._spawn_powerup(block["rect"].center)
                    break # Only hit one block per frame
        
        # Handle lost balls
        for ball in balls_to_remove:
            self.balls.remove(ball)

        if not self.balls:
            self.lives -= 1
            # sfx: lose_life
            if self.lives > 0:
                self._reset_ball()

        return reward

    def _bounce_ball(self, ball, paddle):
        offset = (ball["pos"].x - paddle.centerx) / (paddle.width / 2)
        bounce_angle = offset * (math.pi / 2.5) # Max 72 degrees

        new_vel = pygame.Vector2(math.sin(bounce_angle), -math.cos(bounce_angle))
        ball["vel"] = new_vel * self.BALL_SPEED

    def _reset_ball(self):
        ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        new_ball = {"pos": ball_pos, "vel": pygame.Vector2(0, 0), "attached": True}
        self.balls.append(new_ball)
        self.is_paddle_sticky = False # Sticky effect is one-time use
        self.powerup_timers["sticky"] = 0

    def _create_particles(self, pos, color, count=15):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _spawn_powerup(self, pos):
        powerup_type = self.np_random.choice(["multi_ball", "extended_paddle", "sticky_paddle"])
        self.falling_powerups.append({
            "rect": pygame.Rect(pos[0] - self.POWERUP_SIZE / 2, pos[1] - self.POWERUP_SIZE / 2, self.POWERUP_SIZE, self.POWERUP_SIZE),
            "type": powerup_type
        })
    
    def _update_falling_powerups(self):
        reward = 0
        for p_up in self.falling_powerups[:]:
            p_up["rect"].y += self.POWERUP_SPEED
            if p_up["rect"].colliderect(self.paddle) and self.held_powerup is None:
                self.held_powerup = p_up["type"]
                self.falling_powerups.remove(p_up)
                self.score += 250
                reward += 5
                # sfx: collect_powerup
            elif p_up["rect"].top > self.HEIGHT:
                self.falling_powerups.remove(p_up)
        return reward

    def _activate_powerup(self):
        if self.held_powerup == "multi_ball":
            # sfx: activate_multiball
            if self.balls:
                original_ball = self.balls[0]
                for i in [-1, 1]:
                    new_vel = original_ball["vel"].rotate(i * 25)
                    new_ball = {"pos": original_ball["pos"].copy(), "vel": new_vel, "attached": False}
                    self.balls.append(new_ball)
        elif self.held_powerup == "extended_paddle":
            # sfx: activate_extend
            self.powerup_timers["extended"] = self.POWERUP_DURATION_STEPS
            self.is_paddle_extended = True
        elif self.held_powerup == "sticky_paddle":
            # sfx: activate_sticky
            self.powerup_timers["sticky"] = self.POWERUP_DURATION_STEPS
            self.is_paddle_sticky = True
        
        self.held_powerup = None

    def _update_powerup_timers(self):
        # Extended Paddle
        if self.powerup_timers["extended"] > 0:
            self.powerup_timers["extended"] -= 1
            if self.powerup_timers["extended"] == 0:
                self.is_paddle_extended = False
        
        # Sticky Paddle
        if self.powerup_timers["sticky"] > 0:
            self.powerup_timers["sticky"] -= 1
            if self.powerup_timers["sticky"] == 0:
                self.is_paddle_sticky = False
        
        # Update paddle width based on powerup
        target_width = self.PADDLE_WIDTH * 2 if self.is_paddle_extended else self.PADDLE_WIDTH
        # Smooth transition
        current_width = self.paddle.width
        new_width = current_width + (target_width - current_width) * 0.1
        centerx = self.paddle.centerx
        self.paddle.width = new_width
        self.paddle.centerx = centerx
        
        self.paddle.centerx = np.clip(self.paddle.centerx, self.paddle.width/2, self.WIDTH - self.paddle.width/2)


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
            "lives": self.lives,
            "blocks_left": len(self.blocks),
            "held_powerup": self.held_powerup,
        }

    def _render_game(self):
        self._render_blocks()
        self._render_particles()
        self._render_falling_powerups()
        self._render_paddle()
        self._render_balls()

    def _render_blocks(self):
        for block in self.blocks:
            r = block["rect"]
            c = block["color"]
            # Draw a slightly darker version for a 3D effect
            dark_c = tuple(max(0, val - 40) for val in c)
            pygame.draw.rect(self.screen, dark_c, r.move(2, 2))
            pygame.draw.rect(self.screen, c, r)
            # Add a white highlight
            pygame.draw.rect(self.screen, (255,255,255, 50), r.inflate(-r.width*0.7, -r.height*0.7).move(r.width*0.1, -r.height*0.1), 0, border_radius=2)


    def _render_paddle(self):
        color = self.COLOR_PADDLE
        if self.is_paddle_sticky:
            color = (100, 255, 100) # Green for sticky
        
        # Glow effect
        glow_rect = self.paddle.inflate(6, 6)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, color + (50,), glow_surface.get_rect(), border_radius=6)
        self.screen.blit(glow_surface, glow_rect.topleft)

        pygame.draw.rect(self.screen, color, self.paddle, border_radius=6)

    def _render_balls(self):
        for ball in self.balls:
            pos = (int(ball["pos"].x), int(ball["pos"].y))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS + 3, self.COLOR_BALL + (50,))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS + 3, self.COLOR_BALL + (50,))
            # Ball
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = p["color"] + (alpha,)
            size = int(p["lifespan"] / 5)
            if size > 0:
                pygame.draw.rect(self.screen, color, (p["pos"].x, p["pos"].y, size, size))

    def _render_falling_powerups(self):
        for p_up in self.falling_powerups:
            color = self.RAINBOW_COLORS[self.rainbow_idx]
            pygame.draw.rect(self.screen, (0,0,0), p_up["rect"].inflate(4,4), border_radius=4)
            pygame.draw.rect(self.screen, color, p_up["rect"], border_radius=4)
            # Draw icon based on type
            char = "?"
            if "multi" in p_up["type"]: char = "3"
            if "extend" in p_up["type"]: char = "E"
            if "sticky" in p_up["type"]: char = "S"
            text_surf = self.font_small.render(char, True, (0,0,0))
            self.screen.blit(text_surf, text_surf.get_rect(center=p_up["rect"].center))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score:06}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            life_rect = pygame.Rect(self.WIDTH - (i + 1) * 30, 10, 25, 8)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_rect, border_radius=3)

        # Held Powerup
        if self.held_powerup is not None:
            text = f"POWERUP: {self.held_powerup.replace('_', ' ').upper()}"
            power_surf = self.font_small.render(text, True, self.RAINBOW_COLORS[self.rainbow_idx])
            self.screen.blit(power_surf, power_surf.get_rect(centerx=self.WIDTH/2, y=10))

        # Game Over Message
        if self.game_over:
            msg = "YOU WIN!" if not self.blocks else "GAME OVER"
            color = (0, 255, 0) if not self.blocks else (255, 0, 0)
            end_surf = self.font_large.render(msg, True, color)
            self.screen.blit(end_surf, end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # Un-comment the line below to run with a visible window
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame window for human play ---
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Render to the screen ---
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame uses (width, height), numpy uses (height, width). Transpose is needed.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling and frame rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()