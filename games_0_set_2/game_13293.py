import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:27:44.581553
# Source Brief: brief_03293.md
# Brief Index: 3293
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game objects
class Ball:
    def __init__(self, pos, vel, radius, color):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.trail = []

class Orb:
    def __init__(self, pos, orb_type, radius):
        self.pos = pygame.Vector2(pos)
        self.type = orb_type
        self.radius = radius
        self.base_radius = radius
        self.pulse_phase = random.uniform(0, 2 * math.pi)

class Particle:
    def __init__(self, pos, vel, color, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    game_description = (
        "Control the bounce of three balls to collect green orbs for points and avoid red ones. "
        "Achieve a high score before time runs out."
    )
    user_guide = "Use the arrow keys (↑↓←→) to influence the direction of the balls when they bounce off walls."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds
        self.WIN_SCORE = 80
        self.NUM_ORBS_TARGET = 15
        
        # --- Colors ---
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (30, 35, 60)
        self.COLOR_BALLS = [(0, 255, 255), (255, 0, 255), (255, 255, 0)] # Cyan, Magenta, Yellow
        self.COLOR_GREEN_ORB = (0, 255, 128)
        self.COLOR_RED_ORB = (255, 80, 80)
        self.COLOR_UI_TEXT = (220, 220, 240)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        if self.render_mode == "human":
            self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Bouncing Ball Collector")
        
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls = []
        self.orbs = []
        self.particles = []
        self.bounce_angle_modifier = pygame.Vector2(0, 0)
        self.last_bounce_steps = []
        self.last_sync_reward_step = -100
        self.reward_this_step = 0.0

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._initialize_balls()
        self.orbs = []
        self._spawn_orbs(self.NUM_ORBS_TARGET)
        self.particles = []
        
        self.bounce_angle_modifier = pygame.Vector2(0, 0)
        self.last_bounce_steps = [-1, -1, -1]
        self.last_sync_reward_step = -100
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.reward_this_step = 0.0
        
        if self.game_over:
            # If the game is already over, do nothing but return the final state
            return self._get_observation(), 0.0, True, False, self._get_info()

        self._handle_action(action)
        self._update_game_state()
        
        self.steps += 1
        
        reward = self.reward_this_step
        terminated = self._check_termination()
        truncated = False # This game has a fixed time limit, so it terminates, not truncates.

        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0 # Goal-oriented reward for winning
                # sfx_win()
        
        if self.render_mode == "human":
            self._render_to_window()

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _initialize_balls(self):
        self.balls = []
        ball_radius = 12
        for i in range(3):
            pos = (
                self.np_random.uniform(self.WIDTH * 0.25, self.WIDTH * 0.75),
                self.np_random.uniform(self.HEIGHT * 0.25, self.HEIGHT * 0.75)
            )
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(3, 5)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.balls.append(Ball(pos, vel, ball_radius, self.COLOR_BALLS[i]))
            
    def _handle_action(self, action):
        movement = action[0]
        mod_strength = 2.5
        
        if movement == 1: self.bounce_angle_modifier = pygame.Vector2(0, -mod_strength) # Up
        elif movement == 2: self.bounce_angle_modifier = pygame.Vector2(0, mod_strength) # Down
        elif movement == 3: self.bounce_angle_modifier = pygame.Vector2(-mod_strength, 0) # Left
        elif movement == 4: self.bounce_angle_modifier = pygame.Vector2(mod_strength, 0) # Right
        else: self.bounce_angle_modifier = pygame.Vector2(0, 0) # None
    
    def _update_game_state(self):
        self._update_balls()
        self._update_particles()
        self._manage_orbs()
        self._check_sync_bounce()

    def _update_balls(self):
        if any(b.vel.length_squared() > 0.01 for b in self.balls):
            self.reward_this_step += 0.1

        for i, ball in enumerate(self.balls):
            ball.trail.append(ball.pos.copy())
            if len(ball.trail) > 10:
                ball.trail.pop(0)

            ball.vel *= 0.998 # Air drag
            ball.pos += ball.vel

            # Wall collisions
            bounced = False
            if ball.pos.x - ball.radius < 0:
                ball.pos.x = ball.radius
                ball.vel.x *= -1
                ball.vel += self.bounce_angle_modifier
                bounced = True
            elif ball.pos.x + ball.radius > self.WIDTH:
                ball.pos.x = self.WIDTH - ball.radius
                ball.vel.x *= -1
                ball.vel += self.bounce_angle_modifier
                bounced = True

            if ball.pos.y - ball.radius < 0:
                ball.pos.y = ball.radius
                ball.vel.y *= -1
                ball.vel += self.bounce_angle_modifier
                bounced = True
            elif ball.pos.y + ball.radius > self.HEIGHT:
                ball.pos.y = self.HEIGHT - ball.radius
                ball.vel.y *= -1
                ball.vel += self.bounce_angle_modifier
                bounced = True
            
            if bounced:
                self.last_bounce_steps[i] = self.steps
                # sfx_bounce()

            # Orb collisions
            for orb in self.orbs[:]:
                dist_sq = (ball.pos - orb.pos).length_squared()
                if dist_sq < (ball.radius + orb.radius)**2:
                    self._handle_orb_collection(ball, orb)
                    self.orbs.remove(orb)
                    break
            
            # Speed cap
            max_speed = 15
            if ball.vel.length() > max_speed:
                ball.vel.scale_to_length(max_speed)
    
    def _check_sync_bounce(self):
        sync_window = 2 # frames
        if min(self.last_bounce_steps) > -1 and \
           (max(self.last_bounce_steps) - min(self.last_bounce_steps) <= sync_window) and \
           (self.steps > self.last_sync_reward_step + self.FPS / 2):
            
            self.score += 5
            self.reward_this_step += 5.0
            self.last_sync_reward_step = self.steps
            
            # Spawn a visual indicator for the sync bonus
            center_pos = sum([b.pos for b in self.balls], pygame.Vector2()) / len(self.balls)
            self._spawn_particles(center_pos.x, center_pos.y, (255, 255, 0), 30, speed_mult=1.5)
            # sfx_sync_bonus()
            
    def _handle_orb_collection(self, ball, orb):
        if orb.type == 'green':
            self.score += 10
            self.reward_this_step += 10.0
            self._spawn_particles(orb.pos.x, orb.pos.y, self.COLOR_GREEN_ORB, 20)
            # sfx_collect_good()
        elif orb.type == 'red':
            self.score = max(0, self.score - 5)
            self.reward_this_step -= 5.0
            ball.vel *= 0.5 # Momentum loss
            self._spawn_particles(orb.pos.x, orb.pos.y, self.COLOR_RED_ORB, 20)
            # sfx_collect_bad()

    def _update_particles(self):
        for p in self.particles[:]:
            p.pos += p.vel
            p.lifespan -= 1
            if p.lifespan <= 0:
                self.particles.remove(p)

    def _manage_orbs(self):
        # Pulsate existing orbs
        for orb in self.orbs:
            orb.pulse_phase += 0.1
            orb.radius = orb.base_radius + math.sin(orb.pulse_phase) * 2
        
        # Spawn new orbs if needed
        if len(self.orbs) < self.NUM_ORBS_TARGET:
            self._spawn_orbs(1)

    def _spawn_orbs(self, count):
        for _ in range(count):
            orb_type = 'green' if self.np_random.random() > 0.3 else 'red'
            radius = 10
            
            # Ensure orbs don't spawn too close to balls or other orbs
            while True:
                pos = (self.np_random.uniform(radius, self.WIDTH - radius),
                       self.np_random.uniform(radius, self.HEIGHT - radius))
                
                too_close = False
                for ball in self.balls:
                    if (ball.pos - pos).length() < ball.radius + radius + 50:
                        too_close = True
                        break
                if not too_close:
                    for other_orb in self.orbs:
                        if (other_orb.pos - pos).length() < other_orb.radius + radius + 10:
                            too_close = True
                            break
                if not too_close:
                    break
            
            self.orbs.append(Orb(pos, orb_type, radius))

    def _spawn_particles(self, x, y, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * speed_mult
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append(Particle((x, y), vel, color, lifespan))

    def _get_observation(self):
        self._render_to_surface()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_to_surface(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_orbs()
        self._render_balls()
        self._render_ui()

    def _render_to_window(self):
        self.window.blit(self.screen, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p.lifespan / p.max_lifespan))
            color = (*p.color, alpha)
            pos = (int(p.pos.x), int(p.pos.y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)
    
    def _render_orbs(self):
        for orb in self.orbs:
            pos = (int(orb.pos.x), int(orb.pos.y))
            radius = int(orb.radius)
            color = self.COLOR_GREEN_ORB if orb.type == 'green' else self.COLOR_RED_ORB
            
            # Glow
            glow_radius = int(radius * 1.8)
            glow_alpha = 80
            glow_color = (*color, glow_alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, glow_color)
            
            # Orb body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

    def _render_balls(self):
        for ball in self.balls:
            # Trail
            if len(ball.trail) > 1:
                faded_color = (ball.color[0] // 4, ball.color[1] // 4, ball.color[2] // 4)
                pygame.draw.aalines(self.screen, faded_color, False, ball.trail, 2)

            pos = (int(ball.pos.x), int(ball.pos.y))
            radius = int(ball.radius)
            
            # Glow
            glow_radius = int(radius * 2.5)
            glow_alpha = 100
            glow_color = (*ball.color, glow_alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, glow_color)

            # Ball body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, ball.color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, ball.color)
            
    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_large.render(f"{time_left:.1f}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)

        if self.game_over:
            message = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME'S UP!"
            color = self.COLOR_GREEN_ORB if self.score >= self.WIN_SCORE else self.COLOR_RED_ORB
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left_seconds": max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        }
        
    def _check_termination(self):
        return self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE

    def close(self):
        if self.render_mode == "human":
            pygame.display.quit()
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0] # Default action: no movement
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Map keys to MultiDiscrete action
            # actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                action[0] = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                action[0] = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                action[0] = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                action[0] = 4
            else:
                action[0] = 0
            
            # actions[1]: Space button (0=released, 1=held)
            if keys[pygame.K_SPACE]:
                action[1] = 1
            
            # actions[2]: Shift button (0=released, 1=held)
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

            if terminated:
                print(f"--- Episode Finished ---")
                print(f"Final Score: {info['score']}, Total Steps: {info['steps']}")
                print("Press 'R' to restart.")
    
    env.close()