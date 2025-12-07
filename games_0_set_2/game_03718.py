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


# Helper classes for game objects
class Ball:
    def __init__(self, pos, vel, radius, color, glow_color):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.glow_color = glow_color

class Particle:
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan

class PowerUp:
    def __init__(self, pos, type, size):
        self.pos = pygame.Vector2(pos)
        self.type = type
        self.size = size
        self.rect = pygame.Rect(pos.x - size[0] / 2, pos.y - size[1] / 2, *size)
        self.vel_y = 1.5

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to activate Multi-Ball and shift to activate Wide-Paddle after collecting them."
    )

    game_description = (
        "A fast-paced, retro arcade block breaker. Clear all blocks across three stages before you run out of lives or time!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PADDLE = (50, 150, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BALL_GLOW = (180, 210, 255)
        self.COLOR_POWERUP_WIDE = (50, 255, 150)
        self.COLOR_POWERUP_MULTI = (255, 200, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_GRID = (30, 35, 50)
        self.BLOCK_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255), 
            (255, 255, 80), (255, 80, 255)
        ]

        # --- Fonts ---
        self.font_main = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 60)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.stage = 0
        self.stage_timer = 0
        self.max_steps_per_stage = 1800 # 60 seconds at 30fps

        self.paddle = None
        self.paddle_speed = 8.0
        self.base_paddle_width = 80
        
        self.balls = []
        self.base_ball_speed = 4.0
        
        self.blocks = []
        self.block_size = (56, 20)

        self.particles = []
        self.powerups_on_screen = []
        self.collected_powerups = {}
        self.active_powerup_timers = {}
        self.powerup_drop_chance = 0.2

        # This call to reset() is necessary to initialize game objects
        # before any other methods are called.
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        self.stage = 1
        
        self.balls = []
        self.particles = []
        self.powerups_on_screen = []
        self.collected_powerups = {"wide_paddle": 0, "multi_ball": 0}
        self.active_powerup_timers = {"wide_paddle": 0}
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()
    
    def _setup_stage(self):
        self.stage_timer = self.max_steps_per_stage
        self.paddle = pygame.Rect(
            (self.width - self.base_paddle_width) / 2, 
            self.height - 40, 
            self.base_paddle_width, 
            15
        )
        self.active_powerup_timers["wide_paddle"] = 0
        
        self._spawn_ball(is_reset=True)
        
        self.blocks = []
        cols, rows = 8, 5
        grid_width = cols * (self.block_size[0] + 4)
        start_x = (self.width - grid_width) / 2
        for r in range(rows):
            for c in range(cols):
                color = self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                block_rect = pygame.Rect(
                    start_x + c * (self.block_size[0] + 4),
                    50 + r * (self.block_size[1] + 4),
                    self.block_size[0],
                    self.block_size[1]
                )
                self.blocks.append({"rect": block_rect, "color": color})

    def _spawn_ball(self, is_reset=False, pos=None, vel=None):
        if is_reset:
            self.balls.clear()
        
        if pos is None:
            # FIX: Convert paddle.center (tuple) to a pygame.Vector2 for subtraction
            pos = pygame.Vector2(self.paddle.center) - (0, 80)
        
        if vel is None:
            angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            speed = self.base_ball_speed + (self.stage - 1) * 0.5
            vel = pygame.Vector2(math.sin(angle), -math.cos(angle)) * speed

        self.balls.append(Ball(
            pos, vel, 8, self.COLOR_BALL, self.COLOR_BALL_GLOW
        ))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1
        self.stage_timer -= 1
        
        # --- Update Game Logic ---
        self._handle_input(action)
        self._update_timers()
        
        ball_lost, blocks_broken_count = self._update_balls()
        powerups_collected_count = self._update_powerups()
        
        self._update_particles()
        
        # --- Calculate Reward ---
        reward += 0.001 # Small reward for surviving
        reward += blocks_broken_count * 1.0
        reward += powerups_collected_count * 5.0
        
        # --- Handle State Transitions ---
        life_lost_this_step = False
        if ball_lost:
            self.lives -= 1
            life_lost_this_step = True
            reward -= 20.0
            if self.lives > 0:
                self._spawn_ball(is_reset=False)
            else:
                self.game_over = True

        stage_cleared = False
        if not self.blocks:
            stage_cleared = True
            self.stage += 1
            reward += 100
            if self.stage > 3:
                self.game_over = True
                reward += 300 # Win bonus
            else:
                self._setup_stage()
        
        if self.stage_timer <= 0 and not self.game_over:
            self.game_over = True
            life_lost_this_step = True # Treat as a loss
        
        if self.game_over and life_lost_this_step:
            reward -= 100 # Game over penalty

        terminated = self.game_over
        truncated = False # This environment does not truncate
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 3: # Left
            self.paddle.x -= self.paddle_speed
        elif movement == 4: # Right
            self.paddle.x += self.paddle_speed
        
        self.paddle.x = max(0, min(self.width - self.paddle.width, self.paddle.x))

        if space_held and self.collected_powerups["multi_ball"] > 0:
            self.collected_powerups["multi_ball"] -= 1
            # sfx: powerup_activate_multiball
            if self.balls:
                center_ball_pos = self.balls[0].pos
                self._spawn_ball(pos=center_ball_pos, vel=pygame.Vector2(-3, -3))
                self._spawn_ball(pos=center_ball_pos, vel=pygame.Vector2(3, -3))

        if shift_held and self.collected_powerups["wide_paddle"] > 0:
            self.collected_powerups["wide_paddle"] -= 1
            # sfx: powerup_activate_widepaddle
            self.active_powerup_timers["wide_paddle"] = 300 # 10 seconds

    def _update_timers(self):
        if self.active_powerup_timers["wide_paddle"] > 0:
            self.active_powerup_timers["wide_paddle"] -= 1
            new_width = self.base_paddle_width * 1.5
        else:
            new_width = self.base_paddle_width
        
        center = self.paddle.centerx
        self.paddle.width = new_width
        self.paddle.centerx = center
        self.paddle.x = max(0, min(self.width - self.paddle.width, self.paddle.x))

    def _update_balls(self):
        blocks_broken = 0
        
        for ball in self.balls[:]:
            ball.pos += ball.vel

            # Wall collision
            if ball.pos.x - ball.radius < 0 or ball.pos.x + ball.radius > self.width:
                ball.vel.x *= -1
                ball.pos.x = max(ball.radius, min(self.width - ball.radius, ball.pos.x))
                # sfx: bounce_wall
            if ball.pos.y - ball.radius < 0:
                ball.vel.y *= -1
                ball.pos.y = max(ball.radius, ball.pos.y)
                # sfx: bounce_wall

            # Paddle collision
            if self.paddle.collidepoint(ball.pos.x, ball.pos.y + ball.radius) and ball.vel.y > 0:
                ball.vel.y *= -1
                
                offset = (ball.pos.x - self.paddle.centerx) / (self.paddle.width / 2)
                ball.vel.x += offset * 2.0
                
                # Normalize speed
                speed = self.base_ball_speed + (self.stage - 1) * 0.5
                ball.vel = ball.vel.normalize() * speed
                # sfx: bounce_paddle
                
            # Block collision
            for block in self.blocks[:]:
                if block["rect"].collidepoint(ball.pos):
                    # sfx: break_block
                    blocks_broken += 1
                    
                    # Determine bounce direction
                    dx = ball.pos.x - block["rect"].centerx
                    dy = ball.pos.y - block["rect"].centery
                    if abs(dx / block["rect"].width) > abs(dy / block["rect"].height):
                        ball.vel.x *= -1
                    else:
                        ball.vel.y *= -1
                    
                    self._create_particles(block["rect"].center, block["color"])
                    
                    if self.np_random.random() < self.powerup_drop_chance:
                        ptype = self.np_random.choice(["wide_paddle", "multi_ball"])
                        self.powerups_on_screen.append(PowerUp(pygame.Vector2(block["rect"].center), ptype, (16, 16)))

                    self.blocks.remove(block)
                    self.score += 10
                    break
            
            # Out of bounds
            if ball.pos.y - ball.radius > self.height:
                self.balls.remove(ball)
                # sfx: lose_life
        
        return len(self.balls) == 0 and blocks_broken == 0, blocks_broken

    def _update_powerups(self):
        collected_count = 0
        for pu in self.powerups_on_screen[:]:
            pu.pos.y += pu.vel_y
            pu.rect.top = pu.pos.y
            
            if self.paddle.colliderect(pu.rect):
                # sfx: collect_powerup
                self.collected_powerups[pu.type] = min(3, self.collected_powerups[pu.type] + 1)
                self.powerups_on_screen.remove(pu)
                collected_count += 1
            elif pu.pos.y > self.height:
                self.powerups_on_screen.remove(pu)
        return collected_count

    def _update_particles(self):
        for p in self.particles[:]:
            p.pos += p.vel
            p.vel.y += 0.1 # Gravity
            p.lifespan -= 1
            p.radius = max(0, p.radius - 0.1)
            if p.lifespan <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append(Particle(
                pygame.Vector2(pos), vel, self.np_random.uniform(2, 5), color, self.np_random.integers(20, 40)
            ))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.width, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.height))
        for y in range(0, self.height, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.width, y))

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            
        # Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), int(p.radius), p.color)

        # Powerups
        for pu in self.powerups_on_screen:
            color = self.COLOR_POWERUP_WIDE if pu.type == "wide_paddle" else self.COLOR_POWERUP_MULTI
            pygame.draw.rect(self.screen, color, pu.rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_BALL, pu.rect, width=1, border_radius=4)

        # Balls
        for ball in self.balls:
            # Glow effect
            glow_radius = int(ball.radius * 2.5)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*ball.glow_color, 40), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (int(ball.pos.x - glow_radius), int(ball.pos.y - glow_radius)))
            # Ball
            pygame.gfxdraw.aacircle(self.screen, int(ball.pos.x), int(ball.pos.y), int(ball.radius), ball.color)
            pygame.gfxdraw.filled_circle(self.screen, int(ball.pos.x), int(ball.pos.y), int(ball.radius), ball.color)

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Lives
        lives_surf = self.font_main.render(f"LIVES: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_surf, (self.width - lives_surf.get_width() - 10, 10))
        
        # Stage & Timer
        stage_text = f"STAGE {self.stage}"
        timer_text = f"TIME: {max(0, self.stage_timer // 30)}"
        stage_surf = self.font_main.render(stage_text, True, self.COLOR_UI_TEXT)
        timer_surf = self.font_main.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_surf, (self.width/2 - stage_surf.get_width()/2, 5))
        self.screen.blit(timer_surf, (self.width/2 - timer_surf.get_width()/2, 25))
        
        # Collected Powerups
        if self.collected_powerups["wide_paddle"] > 0:
            pygame.draw.rect(self.screen, self.COLOR_POWERUP_WIDE, (10, self.height - 25, 15, 15), border_radius=3)
            count_surf = self.font_main.render(f"x{self.collected_powerups['wide_paddle']}", True, self.COLOR_UI_TEXT)
            self.screen.blit(count_surf, (30, self.height - 30))
        if self.collected_powerups["multi_ball"] > 0:
            pygame.draw.rect(self.screen, self.COLOR_POWERUP_MULTI, (80, self.height - 25, 15, 15), border_radius=3)
            count_surf = self.font_main.render(f"x{self.collected_powerups['multi_ball']}", True, self.COLOR_UI_TEXT)
            self.screen.blit(count_surf, (100, self.height - 30))
            
        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = not self.blocks and self.stage > 3
            msg = "YOU WIN!" if win_condition else "GAME OVER"
            msg_surf = self.font_large.render(msg, True, self.COLOR_BALL)
            self.screen.blit(msg_surf, (self.width/2 - msg_surf.get_width()/2, self.height/2 - msg_surf.get_height()/2))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set `human_render=True` to see the game window
    human_render = True
    
    if human_render:
        # Re-initialize pygame with the default video driver
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
        pygame.quit()
        pygame.init()
        env = GameEnv(render_mode="human")
        pygame.display.set_caption("Block Breaker")
        game_screen = pygame.display.set_mode((env.width, env.height))
    else:
        env = GameEnv()

    obs, info = env.reset()
    
    terminated = False
    total_reward = 0
    
    # Game loop
    while not terminated:
        # --- Action Mapping for Human Player ---
        action = [0, 0, 0] # Default no-op
        if human_render:
            keys = pygame.key.get_pressed()
            
            movement = 0 # no-op
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
                
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Event Handling & Frame Rate ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
        else: # For non-human mode, step with random actions
            action = env.action_space.sample()

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if human_render:
            # --- Render to Screen ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            game_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30) # 30 FPS
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            if human_render:
                pygame.time.wait(2000) # Pause before closing

    env.close()