
# Generated: 2025-08-28T05:48:43.059812
# Source Brief: brief_02741.md
# Brief Index: 2741

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball. Press shift to use a collected power-up."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block-breaker. Clear all the blocks to win, but don't let the ball fall past your paddle!"
    )

    # Frames auto-advance at 30fps for smooth real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.WIDTH, self.HEIGHT = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.COLOR_BG_TOP = (10, 0, 20)
        self.COLOR_BG_BOTTOM = (40, 0, 60)
        self.COLOR_PADDLE = (255, 255, 0)
        self.COLOR_BALL = (255, 0, 128)
        self.COLOR_UI = (200, 200, 255)
        self.COLOR_BOUNDARY = (100, 80, 150)
        self.BLOCK_COLORS = {
            10: (0, 200, 100),  # Green
            20: (0, 150, 255),  # Blue
            30: (200, 50, 200),  # Magenta
        }
        self.POWERUP_COLORS = {
            "extend": (255, 165, 0),  # Orange
            "multiball": (138, 43, 226),  # BlueViolet
        }
        
        # Game constants
        self.PADDLE_BASE_WIDTH = 80
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_MAX_SPEED = 7
        self.POWERUP_SIZE = 15
        self.POWERUP_SPEED = 2.5
        self.POWERUP_SPAWN_CHANCE = 0.2
        self.POWERUP_DURATION = 300  # 10 seconds at 30fps

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.paddle = None
        self.balls = []
        self.blocks = []
        self.particles = []
        self.powerups = []
        self.ball_attached = True
        self.held_powerup = None
        self.active_powerup = None
        self.active_powerup_timer = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_BASE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_BASE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.balls = []
        self._reset_ball()
        
        self.blocks = []
        block_width, block_height = 58, 20
        for r in range(5):
            for c in range(10):
                points = random.choice(list(self.BLOCK_COLORS.keys()))
                block = {
                    "rect": pygame.Rect(
                        c * (block_width + 2) + 20,
                        r * (block_height + 2) + 50,
                        block_width,
                        block_height
                    ),
                    "points": points,
                    "color": self.BLOCK_COLORS[points]
                }
                self.blocks.append(block)

        self.particles = []
        self.powerups = []
        self.held_powerup = None
        self.active_powerup = None
        self.active_powerup_timer = 0
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_attached = True
        ball_pos = (self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.balls = [{
            "pos": list(ball_pos),
            "vel": [0, 0]
        }]
        # Reset paddle size if extend powerup was active
        if self.active_powerup == "extend":
            self.paddle.width = self.PADDLE_BASE_WIDTH
            self.paddle.centerx = self.paddle.centerx # re-center
            self.active_powerup = None
            self.active_powerup_timer = 0

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1
        
        # 1. Handle Input
        movement = action[0]
        space_press = action[1] == 1
        shift_press = action[2] == 1

        moved = False
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            moved = True
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            moved = True
        
        if moved:
            reward -= 0.01 # Small penalty for movement

        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.WIDTH, self.paddle.right)

        if self.ball_attached and space_press:
            self.ball_attached = False
            initial_vx = self.np_random.uniform(-1, 1)
            self.balls[0]["vel"] = [initial_vx, -self.BALL_MAX_SPEED * 0.8]
            # Sound: Ball launch

        if self.held_powerup and shift_press:
            self._activate_powerup()
            reward += 5.0
            # Sound: Powerup activate

        # 2. Update Game Logic
        self._update_timers()
        self._update_balls(reward)
        self._update_particles()
        self._update_powerups()
        
        # 3. Handle Collisions and generate rewards
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # 4. Check Termination
        terminated = False
        if not self.blocks:
            reward += 100.0  # Win bonus
            terminated = True
            self.game_over = True
        elif self.lives <= 0:
            reward -= 100.0  # Lose penalty
            terminated = True
            self.game_over = True
        elif self.steps >= 2500: # Max steps
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _update_timers(self):
        if self.active_powerup:
            self.active_powerup_timer -= 1
            if self.active_powerup_timer <= 0:
                if self.active_powerup == "extend":
                    self.paddle.inflate_ip(-40, 0)
                    self.paddle.clamp_ip(self.screen.get_rect())
                self.active_powerup = None
                # Sound: Powerup deactivate

    def _update_balls(self, reward):
        if self.ball_attached:
            self.balls[0]["pos"][0] = self.paddle.centerx
            self.balls[0]["pos"][1] = self.paddle.top - self.BALL_RADIUS
        else:
            for ball in self.balls:
                ball["pos"][0] += ball["vel"][0]
                ball["pos"][1] += ball["vel"][1]

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1

    def _update_powerups(self):
        self.powerups = [p for p in self.powerups if p["rect"].top < self.HEIGHT]
        for p in self.powerups:
            p["rect"].y += self.POWERUP_SPEED

    def _handle_collisions(self):
        reward = 0
        balls_to_remove = []
        
        for i, ball in enumerate(self.balls):
            ball_rect = pygame.Rect(ball["pos"][0] - self.BALL_RADIUS, ball["pos"][1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # Walls
            if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
                ball["vel"][0] *= -1
                ball["pos"][0] = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, ball["pos"][0]))
                # Sound: Wall bounce
            if ball_rect.top <= 0:
                ball["vel"][1] *= -1
                # Sound: Wall bounce

            # Paddle
            if ball_rect.colliderect(self.paddle) and ball["vel"][1] > 0:
                ball["vel"][1] *= -1
                offset = (ball_rect.centerx - self.paddle.centerx) / (self.paddle.width / 2)
                ball["vel"][0] = offset * self.BALL_MAX_SPEED
                ball["pos"][1] = self.paddle.top - self.BALL_RADIUS # Prevent sticking
                # Sound: Paddle bounce

            # Blocks
            blocks_to_remove = []
            for j, block in enumerate(self.blocks):
                if ball_rect.colliderect(block["rect"]):
                    blocks_to_remove.append(j)
                    reward += block["points"] / 10.0
                    self.score += block["points"]
                    self._create_particles(block["rect"].center, block["color"])

                    # Simple bounce logic
                    ball["vel"][1] *= -1
                    
                    if self.np_random.random() < self.POWERUP_SPAWN_CHANCE:
                        self._spawn_powerup(block["rect"].center)
                    # Sound: Block break
                    break 
            
            if blocks_to_remove:
                self.blocks = [b for k, b in enumerate(self.blocks) if k not in blocks_to_remove]

            # Bottom edge (lose life)
            if ball_rect.top >= self.HEIGHT:
                balls_to_remove.append(i)

        if balls_to_remove:
            self.balls = [b for i, b in enumerate(self.balls) if i not in balls_to_remove]
            if not self.balls:
                self.lives -= 1
                if self.lives > 0:
                    self._reset_ball()
                    # Sound: Lose life
                else:
                    self.game_over = True
                    # Sound: Game over

        # Paddle collecting powerups
        for powerup in self.powerups[:]:
            if self.paddle.colliderect(powerup["rect"]):
                if not self.held_powerup and not self.active_powerup:
                    self.held_powerup = powerup["type"]
                self.powerups.remove(powerup)
                # Sound: Powerup collect

        return reward

    def _activate_powerup(self):
        if self.held_powerup == "extend":
            self.active_powerup = "extend"
            self.active_powerup_timer = self.POWERUP_DURATION
            self.paddle.inflate_ip(40, 0)
            self.paddle.clamp_ip(self.screen.get_rect())
        elif self.held_powerup == "multiball":
            if not self.ball_attached and self.balls:
                original_ball = self.balls[0]
                for _ in range(2):
                    new_vel = [
                        original_ball["vel"][0] + self.np_random.uniform(-1, 1),
                        original_ball["vel"][1] + self.np_random.uniform(-1, 0)
                    ]
                    # Clamp speed
                    speed = math.hypot(*new_vel)
                    if speed > self.BALL_MAX_SPEED:
                        scale = self.BALL_MAX_SPEED / speed
                        new_vel = [v * scale for v in new_vel]
                    
                    self.balls.append({
                        "pos": list(original_ball["pos"]),
                        "vel": new_vel
                    })
        self.held_powerup = None

    def _spawn_powerup(self, pos):
        ptype = self.np_random.choice(list(self.POWERUP_COLORS.keys()))
        self.powerups.append({
            "rect": pygame.Rect(pos[0] - self.POWERUP_SIZE / 2, pos[1], self.POWERUP_SIZE, self.POWERUP_SIZE),
            "type": ptype,
            "color": self.POWERUP_COLORS[ptype]
        })

    def _create_particles(self, pos, color):
        for _ in range(15):
            self.particles.append({
                "pos": list(pos),
                "vel": [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)],
                "life": self.np_random.integers(10, 20),
                "color": color
            })

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block["color"]), block["rect"], 2)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(p["life"] * 15)))
            size = max(1, int(p["life"] * 0.2))
            color = (*p["color"], alpha)
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect())
            self.screen.blit(s, (int(p["pos"][0] - size), int(p["pos"][1] - size)))

        # Powerups
        for powerup in self.powerups:
            pygame.draw.rect(self.screen, powerup["color"], powerup["rect"], border_radius=3)
            pygame.draw.rect(self.screen, (255,255,255), powerup["rect"], 2, border_radius=3)

        # Paddle
        glow_color = (*self.COLOR_PADDLE, 50)
        glow_rect = self.paddle.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=8)
        self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Balls
        for ball in self.balls:
            pos = (int(ball["pos"][0]), int(ball["pos"][1]))
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS + 4, (*self.COLOR_BALL, 50))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS + 4, (*self.COLOR_BALL, 50))
            # Main ball
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_surf, (20, 10))

        # Lives
        lives_surf = self.font_main.render(f"LIVES: {self.lives}", True, self.COLOR_UI)
        self.screen.blit(lives_surf, (self.WIDTH - lives_surf.get_width() - 20, 10))
        
        # Held Powerup
        if self.held_powerup:
            powerup_text = self.font_small.render("POWERUP READY (SHIFT)", True, self.COLOR_UI)
            text_rect = powerup_text.get_rect(center=(self.WIDTH/2, 25))
            self.screen.blit(powerup_text, text_rect)
            
            # Icon
            color = self.POWERUP_COLORS[self.held_powerup]
            icon_rect = pygame.Rect(0,0,20,20)
            icon_rect.center = (self.WIDTH/2, text_rect.bottom + 15)
            pygame.draw.rect(self.screen, color, icon_rect, border_radius=3)
            pygame.draw.rect(self.screen, (255,255,255), icon_rect, 2, border_radius=3)
            
        # Active Powerup Timer
        if self.active_powerup:
            bar_width = 100
            bar_height = 10
            progress = self.active_powerup_timer / self.POWERUP_DURATION
            
            back_rect = pygame.Rect(0,0,bar_width, bar_height)
            back_rect.center = (self.WIDTH/2, self.HEIGHT - 20)
            
            fill_rect = pygame.Rect(back_rect.topleft, (int(bar_width * progress), bar_height))
            
            pygame.draw.rect(self.screen, (50,50,50), back_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.POWERUP_COLORS[self.active_powerup], fill_rect, border_radius=3)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_left": len(self.blocks),
            "balls_in_play": len(self.balls) if not self.ball_attached else 0,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    
    terminated = False
    clock = pygame.time.Clock()
    
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Match the intended framerate
        
    print(f"Game Over! Final Score: {info['score']}")
    env.close()