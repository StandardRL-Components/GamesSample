
# Generated: 2025-08-28T05:23:34.982459
# Source Brief: brief_05562.md
# Brief Index: 5562

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker where risk-taking is rewarded."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_PADDLE = (0, 200, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_POWERUP = (255, 215, 0)
    COLOR_TEXT = (240, 240, 240)
    BLOCK_COLORS = {
        10: (0, 220, 100),  # Green
        20: (80, 150, 255),  # Blue
        30: (255, 80, 80),   # Red
    }
    
    # Dimensions & Speeds
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PADDLE_WIDTH_INITIAL = 100
    PADDLE_HEIGHT = 12
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    BALL_SPEED_INITIAL = 5
    BALL_SPEED_INCREMENT = 0.2
    MAX_EPISODE_STEPS = 1000
    POWERUP_CHANCE = 0.2
    POWERUP_SPEED = 2
    POWERUP_DURATION = 300 # 10 seconds at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)
        
        self.last_space_state = 0
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH_INITIAL) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH_INITIAL,
            self.PADDLE_HEIGHT
        )
        
        self.balls = []
        self._spawn_ball(attached=True)

        self.blocks = self._generate_blocks()
        self.blocks_destroyed_count = 0
        
        self.particles = []
        self.powerups = []
        self.active_powerups = {}

        self.last_space_state = 0
        
        return self._get_observation(), self._get_info()

    def _generate_blocks(self):
        blocks = []
        rows = 5
        cols = 12
        block_width = 50
        block_height = 20
        gap = 4
        top_offset = 50
        side_offset = (self.SCREEN_WIDTH - (cols * (block_width + gap) - gap)) / 2
        
        for r in range(rows):
            for c in range(cols):
                points = [30, 30, 20, 20, 10][r]
                color = self.BLOCK_COLORS[points]
                rect = pygame.Rect(
                    side_offset + c * (block_width + gap),
                    top_offset + r * (block_height + gap),
                    block_width,
                    block_height
                )
                blocks.append({"rect": rect, "points": points, "color": color})
        return blocks

    def _spawn_ball(self, attached=False, pos=None, vel=None):
        if pos is None:
            pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        if vel is None:
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            vel = [math.cos(angle) * self.BALL_SPEED_INITIAL, math.sin(angle) * self.BALL_SPEED_INITIAL]

        ball = {
            "pos": np.array(pos, dtype=float),
            "vel": np.array(vel, dtype=float),
            "attached": attached,
            "trail": [],
        }
        self.balls.append(ball)
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        reward = 0.0
        
        self._handle_input(movement, space_held)
        self._update_powerups()
        
        ball_reward, terminated_by_ball_loss = self._update_balls()
        reward += ball_reward

        self._update_particles()
        
        if any(not b['attached'] for b in self.balls):
            reward += 0.1

        terminated = self.game_over or terminated_by_ball_loss or self.steps >= self.MAX_EPISODE_STEPS
        
        if not self.blocks: # Win condition
            reward = 100.0
            terminated = True
        
        if self.lives <= 0 and not self.game_over: # Lose condition
            reward = -100.0
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Paddle Movement
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.paddle.width)

        # Ball Launch
        if space_held and not self.last_space_state:
            for ball in self.balls:
                if ball["attached"]:
                    ball["attached"] = False
                    # sfx: launch_ball.wav
        self.last_space_state = space_held

    def _update_balls(self):
        reward = 0
        
        balls_to_remove = []
        for i, ball in enumerate(self.balls):
            if ball["attached"]:
                ball["pos"][0] = self.paddle.centerx
                ball["pos"][1] = self.paddle.top - self.BALL_RADIUS
                continue

            # Update trail
            ball["trail"].append(tuple(ball["pos"]))
            if len(ball["trail"]) > 10:
                ball["trail"].pop(0)

            ball["pos"] += ball["vel"]

            # Wall collisions
            if ball["pos"][0] <= self.BALL_RADIUS or ball["pos"][0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
                ball["vel"][0] *= -1
                ball["pos"][0] = np.clip(ball["pos"][0], self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
                # sfx: bounce_wall.wav
            if ball["pos"][1] <= self.BALL_RADIUS:
                ball["vel"][1] *= -1
                ball["pos"][1] = self.BALL_RADIUS
                # sfx: bounce_wall.wav

            # Paddle collision
            if self.paddle.collidepoint(ball["pos"]) and ball["vel"][1] > 0:
                ball["vel"][1] *= -1
                
                # Influence horizontal velocity
                offset = (ball["pos"][0] - self.paddle.centerx) / (self.paddle.width / 2)
                ball["vel"][0] += offset * 2
                
                # Normalize speed
                speed = np.linalg.norm(ball["vel"])
                ball["vel"] = ball["vel"] / speed * self._get_current_ball_speed()
                
                ball["pos"][1] = self.paddle.top - self.BALL_RADIUS
                # sfx: bounce_paddle.wav
            
            # Block collisions
            hit_block = False
            for block in self.blocks[:]:
                if pygame.Rect(*block["rect"]).collidepoint(ball["pos"]):
                    # sfx: break_block.wav
                    self.blocks.remove(block)
                    hit_block = True
                    self.score += block["points"]
                    reward += block["points"] / 10.0
                    self.blocks_destroyed_count += 1
                    
                    self._create_particles(pygame.Rect(*block["rect"]).center, block["color"])
                    
                    if self.np_random.random() < self.POWERUP_CHANCE:
                        self._spawn_powerup(pygame.Rect(*block["rect"]).center)
                    
                    # Determine bounce direction
                    ball_rect = pygame.Rect(ball["pos"][0] - self.BALL_RADIUS, ball["pos"][1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
                    block_rect = pygame.Rect(*block["rect"])
                    
                    if ball_rect.clipline(block_rect.topleft, block_rect.topright) or ball_rect.clipline(block_rect.bottomleft, block_rect.bottomright):
                        ball["vel"][1] *= -1
                    else:
                        ball["vel"][0] *= -1
                    break
            
            # Bottom wall (lose ball)
            if ball["pos"][1] >= self.SCREEN_HEIGHT:
                balls_to_remove.append(i)
        
        # Remove lost balls
        for i in sorted(balls_to_remove, reverse=True):
            del self.balls[i]
            # sfx: lose_ball.wav

        if not self.balls:
            self.lives -= 1
            if self.lives > 0:
                self._spawn_ball(attached=True)
            else:
                return reward, True # Terminated by ball loss

        return reward, False

    def _get_current_ball_speed(self):
        return self.BALL_SPEED_INITIAL + (self.blocks_destroyed_count // 5) * self.BALL_SPEED_INCREMENT

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                "pos": list(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": color,
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _spawn_powerup(self, pos):
        powerup_type = self.np_random.choice(['long_paddle', 'multi_ball'])
        self.powerups.append({
            "pos": list(pos),
            "rect": pygame.Rect(pos[0]-10, pos[1]-10, 20, 20),
            "type": powerup_type
        })

    def _update_powerups(self):
        # Update falling powerups
        for p in self.powerups[:]:
            p["pos"][1] += self.POWERUP_SPEED
            p["rect"].y = p["pos"][1]
            if self.paddle.colliderect(p["rect"]):
                self._activate_powerup(p["type"])
                self.powerups.remove(p)
                self.score += 25
                # sfx: powerup_collect.wav
            elif p["pos"][1] > self.SCREEN_HEIGHT:
                self.powerups.remove(p)
        
        # Update active powerup effects
        for effect, timer in list(self.active_powerups.items()):
            self.active_powerups[effect] -= 1
            if self.active_powerups[effect] <= 0:
                del self.active_powerups[effect]
                self._deactivate_powerup(effect)

    def _activate_powerup(self, p_type):
        if p_type == 'long_paddle':
            self.active_powerups['long_paddle'] = self.POWERUP_DURATION
            self.paddle.width = self.PADDLE_WIDTH_INITIAL * 1.5
            self.paddle.x -= self.PADDLE_WIDTH_INITIAL * 0.25
        elif p_type == 'multi_ball':
            if self.balls:
                original_ball = self.balls[0]
                if not original_ball['attached']:
                    for i in range(2):
                        angle_offset = self.np_random.uniform(-0.5, 0.5)
                        new_vel = np.array(original_ball['vel'])
                        c, s = math.cos(angle_offset), math.sin(angle_offset)
                        rot_matrix = np.array([[c, -s], [s, c]])
                        new_vel = rot_matrix @ new_vel
                        self._spawn_ball(pos=original_ball['pos'], vel=new_vel)

    def _deactivate_powerup(self, p_type):
        if p_type == 'long_paddle':
            self.paddle.x += (self.paddle.width - self.PADDLE_WIDTH_INITIAL) / 2
            self.paddle.width = self.PADDLE_WIDTH_INITIAL

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p["pos"][0]-2), int(p["pos"][1]-2)))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
        
        # Powerups
        for p in self.powerups:
            pygame.gfxdraw.box(self.screen, p["rect"], (*self.COLOR_POWERUP, 180))
            pygame.gfxdraw.rectangle(self.screen, p["rect"], self.COLOR_POWERUP)
            
        # Ball Trails and Balls
        for ball in self.balls:
            for i, pos in enumerate(ball["trail"]):
                alpha = int(255 * (i / len(ball["trail"])))
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.BALL_RADIUS, (*self.COLOR_BALL, alpha // 4))
            pygame.gfxdraw.aacircle(self.screen, int(ball["pos"][0]), int(ball["pos"][1]), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, int(ball["pos"][0]), int(ball["pos"][1]), self.BALL_RADIUS, self.COLOR_BALL)
            
        # Paddle
        is_long = 'long_paddle' in self.active_powerups
        color = (255, 100, 255) if is_long else self.COLOR_PADDLE
        pygame.draw.rect(self.screen, color, self.paddle, border_radius=4)
        
        # UI
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        for i in range(self.lives):
            pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH - 20 - i*20, 20, self.BALL_RADIUS-2, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH - 20 - i*20, 20, self.BALL_RADIUS-2, self.COLOR_BALL)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    # Requires pygame to be installed with display drivers
    import os
    if os.environ.get("SDL_VIDEODRIVER", "") == "dummy":
        print("Cannot run interactive test in a headless environment. Skipping.")
    else:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        done = False
        
        # Create a display window
        pygame.display.set_caption("Block Breaker Gym Env")
        display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        clock = pygame.time.Clock()

        while not done:
            # --- Human Input ---
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- Rendering ---
            # The observation is already the rendered frame, so we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            clock.tick(30) # Run at 30 FPS

        print(f"Game Over! Final Info: {info}")
        env.close()