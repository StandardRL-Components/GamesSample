import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:46:02.320588
# Source Brief: brief_02473.md
# Brief Index: 2473
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Ball:
    """Helper class for ball properties."""
    def __init__(self, pos, vel, radius, color, glow_color):
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(vel, dtype=np.float64)
        self.radius = float(radius)
        self.color = color
        self.glow_color = glow_color

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a trio of colored balls, collecting green orbs for points. "
        "Red orbs give more points but shrink all balls. Keep your balls large enough to stay in the game!"
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the red ball. Hold space to also move the green ball, "
        "and hold shift to also move the blue ball."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    GAME_AREA_WIDTH, GAME_AREA_HEIGHT = 400, 400
    GAME_AREA_X_OFFSET = (WIDTH - GAME_AREA_WIDTH) // 2
    GAME_AREA_Y_OFFSET = (HEIGHT - GAME_AREA_HEIGHT) // 2
    
    COLOR_BG = (15, 18, 26)
    COLOR_BOUNDARY = (200, 200, 220)
    COLOR_TEXT = (240, 240, 255)
    
    BALL_COLORS = [
        ((255, 80, 80), (255, 80, 80, 60)),    # Red
        ((80, 255, 80), (80, 255, 80, 60)),    # Green
        ((80, 120, 255), (80, 120, 255, 60))  # Blue
    ]
    ORB_GREEN_COLOR = ((0, 255, 150), (0, 255, 150, 80))
    ORB_RED_COLOR = ((255, 70, 70), (255, 70, 70, 80))

    FONT_SIZE_UI = 28
    FONT_SIZE_BALL = 14

    MAX_STEPS = 1800 # ~30 seconds at 60fps
    WIN_SCORE = 25
    
    INITIAL_BALL_RADIUS = 20.0
    MIN_BALL_RADIUS = 3.0
    ORB_RADIUS = 8.0
    NUM_GREEN_ORBS = 5
    NUM_RED_ORBS = 2
    
    FRICTION = 0.985
    ACCELERATION = 0.4
    
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
        self.font_ui = pygame.font.SysFont("Consolas", self.FONT_SIZE_UI, bold=True)
        self.font_ball = pygame.font.SysFont("Consolas", self.FONT_SIZE_BALL)
        
        self.render_mode = render_mode

        self.balls = []
        self.orbs = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        self._initialize_balls()
        self._initialize_orbs()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        green_active = action[1] == 1
        blue_active = action[2] == 1
        
        reward = 0.01 # Survival reward

        # 1. Apply actions
        self._apply_player_action(movement, green_active, blue_active)

        # 2. Update physics
        self._update_ball_physics()
        self._handle_wall_collisions()
        self._handle_ball_collisions()
        
        # 3. Handle game logic
        collected_reward = self._handle_orb_collisions()
        reward += collected_reward

        # 4. Update step counter and check for termination
        self.steps += 1
        terminated = self._check_termination()
        truncated = False # This environment does not truncate
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0 # Win bonus
            elif self._are_all_balls_lost():
                reward -= 100.0 # Lose penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _initialize_balls(self):
        self.balls = []
        start_positions = [
            (self.GAME_AREA_WIDTH * 0.25, self.GAME_AREA_HEIGHT * 0.5),
            (self.GAME_AREA_WIDTH * 0.50, self.GAME_AREA_HEIGHT * 0.5),
            (self.GAME_AREA_WIDTH * 0.75, self.GAME_AREA_HEIGHT * 0.5)
        ]
        for i in range(3):
            pos = start_positions[i]
            vel = self.np_random.uniform(-1, 1, size=2)
            ball = Ball(pos, vel, self.INITIAL_BALL_RADIUS, self.BALL_COLORS[i][0], self.BALL_COLORS[i][1])
            self.balls.append(ball)
            
    def _initialize_orbs(self):
        self.orbs = []
        for _ in range(self.NUM_GREEN_ORBS):
            self._spawn_orb('green')
        for _ in range(self.NUM_RED_ORBS):
            self._spawn_orb('red')

    def _spawn_orb(self, orb_type):
        while True:
            pos = self.np_random.uniform(
                low=[self.ORB_RADIUS, self.ORB_RADIUS],
                high=[self.GAME_AREA_WIDTH - self.ORB_RADIUS, self.GAME_AREA_HEIGHT - self.ORB_RADIUS]
            )
            # Ensure it doesn't spawn on top of another orb
            is_overlapping = False
            for other in self.orbs:
                if np.linalg.norm(pos - other['pos']) < self.ORB_RADIUS * 2.5:
                    is_overlapping = True
                    break
            if not is_overlapping:
                break
        
        color, glow = self.ORB_GREEN_COLOR if orb_type == 'green' else self.ORB_RED_COLOR
        self.orbs.append({'pos': pos, 'type': orb_type, 'color': color, 'glow': glow})

    def _apply_player_action(self, movement, green_active, blue_active):
        if movement == 0: return

        force = np.zeros(2)
        if movement == 1: force[1] = -self.ACCELERATION # Up
        elif movement == 2: force[1] = self.ACCELERATION # Down
        elif movement == 3: force[0] = -self.ACCELERATION # Left
        elif movement == 4: force[0] = self.ACCELERATION # Right

        # Red ball is always affected
        if self.balls[0].radius > self.MIN_BALL_RADIUS:
            self.balls[0].vel += force
        # Green ball if space is held
        if green_active and self.balls[1].radius > self.MIN_BALL_RADIUS:
            self.balls[1].vel += force
        # Blue ball if shift is held
        if blue_active and self.balls[2].radius > self.MIN_BALL_RADIUS:
            self.balls[2].vel += force

    def _update_ball_physics(self):
        for ball in self.balls:
            if ball.radius > self.MIN_BALL_RADIUS:
                ball.vel *= self.FRICTION
                ball.pos += ball.vel

    def _handle_wall_collisions(self):
        for ball in self.balls:
            if ball.radius <= self.MIN_BALL_RADIUS: continue
            
            # Left wall
            if ball.pos[0] - ball.radius < 0:
                ball.pos[0] = ball.radius
                ball.vel[0] *= -0.9 # Lose some energy on bounce
            # Right wall
            if ball.pos[0] + ball.radius > self.GAME_AREA_WIDTH:
                ball.pos[0] = self.GAME_AREA_WIDTH - ball.radius
                ball.vel[0] *= -0.9
            # Top wall
            if ball.pos[1] - ball.radius < 0:
                ball.pos[1] = ball.radius
                ball.vel[1] *= -0.9
            # Bottom wall
            if ball.pos[1] + ball.radius > self.GAME_AREA_HEIGHT:
                ball.pos[1] = self.GAME_AREA_HEIGHT - ball.radius
                ball.vel[1] *= -0.9

    def _handle_ball_collisions(self):
        for i in range(len(self.balls)):
            for j in range(i + 1, len(self.balls)):
                b1, b2 = self.balls[i], self.balls[j]
                if b1.radius <= self.MIN_BALL_RADIUS or b2.radius <= self.MIN_BALL_RADIUS:
                    continue

                dist_vec = b1.pos - b2.pos
                dist = np.linalg.norm(dist_vec)
                min_dist = b1.radius + b2.radius

                if dist < min_dist:
                    # Resolve overlap
                    overlap = (min_dist - dist) / 2
                    if dist > 0:
                        b1.pos += (dist_vec / dist) * overlap
                        b2.pos -= (dist_vec / dist) * overlap
                    else: # Balls are at the exact same spot
                        b1.pos[0] += overlap
                        b2.pos[0] -= overlap

                    dist_vec = b1.pos - b2.pos
                    dist = np.linalg.norm(dist_vec)
                    
                    # Elastic collision
                    normal = dist_vec / dist
                    tangent = np.array([-normal[1], normal[0]])
                    
                    v1n = np.dot(b1.vel, normal)
                    v1t = np.dot(b1.vel, tangent)
                    v2n = np.dot(b2.vel, normal)
                    v2t = np.dot(b2.vel, tangent)

                    m1 = b1.radius ** 2
                    m2 = b2.radius ** 2

                    new_v1n = (v1n * (m1 - m2) + 2 * m2 * v2n) / (m1 + m2)
                    new_v2n = (v2n * (m2 - m1) + 2 * m1 * v1n) / (m1 + m2)

                    b1.vel = new_v1n * normal + v1t * tangent
                    b2.vel = new_v2n * normal + v2t * tangent

    def _handle_orb_collisions(self):
        collected_reward = 0
        orbs_to_remove = []
        for i, orb in enumerate(self.orbs):
            for ball in self.balls:
                if ball.radius <= self.MIN_BALL_RADIUS: continue
                
                dist = np.linalg.norm(ball.pos - orb['pos'])
                if dist < ball.radius + self.ORB_RADIUS:
                    if orb['type'] == 'green':
                        self.score += 1
                        collected_reward += 1.0
                    elif orb['type'] == 'red':
                        self.score += 5
                        collected_reward += 5.0
                        for b in self.balls:
                            b.radius = max(self.MIN_BALL_RADIUS, b.radius * 0.90)
                    
                    orbs_to_remove.append(i)
                    break 
        
        if orbs_to_remove:
            for i in sorted(list(set(orbs_to_remove)), reverse=True):
                orb_type = self.orbs.pop(i)['type']
                self._spawn_orb(orb_type)
        
        return collected_reward

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if self.score >= self.WIN_SCORE:
            return True
        if self._are_all_balls_lost():
            return True
        return False

    def _are_all_balls_lost(self):
        return all(ball.radius <= self.MIN_BALL_RADIUS for ball in self.balls)

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
            "ball_radii": [b.radius for b in self.balls]
        }
        
    def _render_game(self):
        boundary_rect = pygame.Rect(self.GAME_AREA_X_OFFSET - 2, self.GAME_AREA_Y_OFFSET - 2, 
                                    self.GAME_AREA_WIDTH + 4, self.GAME_AREA_HEIGHT + 4)
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, boundary_rect, 2, 8)
        
        game_surface = self.screen.subsurface(pygame.Rect(self.GAME_AREA_X_OFFSET, self.GAME_AREA_Y_OFFSET,
                                                          self.GAME_AREA_WIDTH, self.GAME_AREA_HEIGHT))
        
        for orb in self.orbs:
            x, y = int(orb['pos'][0]), int(orb['pos'][1])
            pygame.gfxdraw.filled_circle(game_surface, x, y, int(self.ORB_RADIUS * 1.5), orb['glow'])
            pygame.gfxdraw.aacircle(game_surface, x, y, int(self.ORB_RADIUS), orb['color'])
            pygame.gfxdraw.filled_circle(game_surface, x, y, int(self.ORB_RADIUS), orb['color'])
        
        for ball in self.balls:
            if ball.radius <= self.MIN_BALL_RADIUS: continue
            
            r = int(ball.radius)
            x, y = int(ball.pos[0]), int(ball.pos[1])
            
            pygame.gfxdraw.filled_circle(game_surface, x, y, int(r * 1.3), ball.glow_color)
            pygame.gfxdraw.aacircle(game_surface, x, y, r, ball.color)
            pygame.gfxdraw.filled_circle(game_surface, x, y, r, ball.color)
            
            radius_text = self.font_ball.render(f"{ball.radius:.0f}", True, self.COLOR_TEXT)
            text_rect = radius_text.get_rect(center=(x, y))
            game_surface.blit(radius_text, text_rect)

    def _render_ui(self):
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))
        
        time_left = (self.MAX_STEPS - self.steps) / 60.0
        time_surf = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        time_rect = time_surf.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(time_surf, time_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Ball Juggler")
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if terminated:
            font = pygame.font.SysFont("Consolas", 50, bold=True)
            text = "GAME OVER"
            if info['score'] >= GameEnv.WIN_SCORE:
                text = "YOU WIN!"
            
            text_surf = font.render(text, True, (255, 255, 0))
            text_rect = text_surf.get_rect(center=(GameEnv.WIDTH/2, GameEnv.HEIGHT/2))
            screen.blit(text_surf, text_rect)
            
            font_small = pygame.font.SysFont("Consolas", 20)
            reset_surf = font_small.render("Press 'R' to restart", True, GameEnv.COLOR_TEXT)
            reset_rect = reset_surf.get_rect(center=(GameEnv.WIDTH/2, GameEnv.HEIGHT/2 + 50))
            screen.blit(reset_surf, reset_rect)

            pygame.display.flip()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
            continue

        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60)

    env.close()