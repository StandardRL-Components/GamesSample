import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:53:09.204695
# Source Brief: brief_00752.md
# Brief Index: 752
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from itertools import combinations

class GameEnv(gym.Env):
    """
    Gymnasium environment for a neon-themed arcade game.
    The player controls the launch angle of two balls and launches them.
    A third ball launches automatically.
    Balls bounce off the walls of a heptagon arena, creating expanding force fields.
    The goal is to score points by making these force fields overlap.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Launch balls into a heptagonal arena. Score points by causing the force fields from wall impacts to overlap."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to aim. Press space to launch the red ball and shift to launch the green ball."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    
    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_WALL = (100, 110, 130)
    COLOR_TEXT = (240, 240, 255)
    COLOR_AIMER = (255, 255, 255, 150)
    
    BALL_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
    ]
    
    # Game Parameters
    ARENA_RADIUS = 180
    WALL_THICKNESS = 4
    BALL_RADIUS = 8
    BALL_SPEEDS = [150, 225, 300] # px/sec
    FORCE_FIELD_DURATION = 1.5 # seconds
    FORCE_FIELD_MAX_RADIUS = 120
    PARTICLE_LIFESPAN = 0.6 # seconds
    
    GAME_DURATION_SECONDS = 60.0
    WIN_SCORE = 500
    
    # Reward structure
    REWARD_WALL_HIT = 1.0
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0
    REWARD_OVERLAP_SCALAR = 0.02
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.time_remaining = 0.0
        self.game_over = False
        
        self.balls = []
        self.force_fields = []
        self.particles = []
        
        self.launch_angle = 0.0
        self.launch_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.blue_ball_launch_timer = 0.0
        
        # Pre-calculate arena geometry
        self._calculate_heptagon_walls()
        
        # Initialize state
        # self.reset() is called by the environment wrapper

    def _calculate_heptagon_walls(self):
        self.walls = []
        self.wall_normals = []
        center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        
        points = []
        for i in range(7):
            angle = (i / 7) * 2 * math.pi + (math.pi / 2) # Rotated for flat bottom
            x = center_x + self.ARENA_RADIUS * math.cos(angle)
            y = center_y + self.ARENA_RADIUS * math.sin(angle)
            points.append(pygame.Vector2(x, y))

        for i in range(7):
            p1 = points[i]
            p2 = points[(i + 1) % 7]
            self.walls.append((p1, p2))
            
            edge = p2 - p1
            normal = pygame.Vector2(-edge.y, edge.x).normalize()
            self.wall_normals.append(normal)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.time_remaining = self.GAME_DURATION_SECONDS
        self.game_over = False
        
        self.balls = [
            {'pos': pygame.Vector2(self.launch_pos), 'vel': pygame.Vector2(0, 0), 'color': self.BALL_COLORS[0], 'speed': self.BALL_SPEEDS[0], 'state': 'ready'},
            {'pos': pygame.Vector2(self.launch_pos), 'vel': pygame.Vector2(0, 0), 'color': self.BALL_COLORS[1], 'speed': self.BALL_SPEEDS[1], 'state': 'ready'},
            {'pos': pygame.Vector2(self.launch_pos), 'vel': pygame.Vector2(0, 0), 'color': self.BALL_COLORS[2], 'speed': self.BALL_SPEEDS[2], 'state': 'ready'},
        ]
        
        self.force_fields = []
        self.particles = []
        
        self.launch_angle = self.np_random.uniform(0, 2 * math.pi)
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.blue_ball_launch_timer = 2.0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # --- Time and Step Management ---
        dt = 1.0 / self.FPS
        self.steps += 1
        self.time_remaining = max(0.0, self.time_remaining - dt)
        
        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Adjust launch angle
        angle_speed = 2.0 * math.pi / self.FPS # One rotation per second
        if movement == 1: # Up
            self.launch_angle -= angle_speed * dt
        elif movement == 2: # Down
            self.launch_angle += angle_speed * dt
        # Left/Right could also be used for angle, here mapped to same as Up/Down for simplicity
        elif movement == 3: # Left
            self.launch_angle -= angle_speed * dt
        elif movement == 4: # Right
            self.launch_angle += angle_speed * dt
        self.launch_angle %= (2 * math.pi)

        # Launch balls on button press (rising edge)
        if space_held and not self.prev_space_held:
            self._launch_ball(0) # Launch Red Ball
        if shift_held and not self.prev_shift_held:
            self._launch_ball(1) # Launch Green Ball
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # Automatic blue ball launch
        self.blue_ball_launch_timer -= dt
        if self.blue_ball_launch_timer <= 0:
            self._launch_ball(2, use_random_angle=True) # Launch Blue Ball
            self.blue_ball_launch_timer = self.np_random.uniform(1.5, 2.5)

        # --- Game Logic Update ---
        reward = 0.0
        
        reward += self._update_balls(dt)
        self._update_force_fields(dt)
        self._update_particles(dt)
        
        # Calculate overlap reward
        if len(self.force_fields) > 1:
            for f1, f2 in combinations(self.force_fields, 2):
                dist = f1['pos'].distance_to(f2['pos'])
                overlap_dist = f1['radius'] + f2['radius'] - dist
                if overlap_dist > 0:
                    # Reward is proportional to the depth of overlap
                    reward += overlap_dist * self.REWARD_OVERLAP_SCALAR * dt

        self.score += reward

        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = False # No truncation condition other than termination
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += self.REWARD_WIN
            else: # Time ran out
                reward += self.REWARD_LOSS
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _launch_ball(self, ball_index, use_random_angle=False):
        ball = self.balls[ball_index]
        if ball['state'] == 'ready':
            ball['state'] = 'active'
            angle = self.np_random.uniform(0, 2 * math.pi) if use_random_angle else self.launch_angle
            ball['vel'] = pygame.Vector2(math.cos(angle), math.sin(angle)) * ball['speed']
            ball['pos'] = pygame.Vector2(self.launch_pos)
            # sfx: launch_sound

    def _update_balls(self, dt):
        wall_hit_reward = 0.0
        for ball in self.balls:
            if ball['state'] == 'active':
                new_pos = ball['pos'] + ball['vel'] * dt
                
                # Wall collision detection and response
                for i, (p1, p2) in enumerate(self.walls):
                    # Simple circle-line segment collision
                    line_vec = p2 - p1
                    point_vec = new_pos - p1
                    line_len_sq = line_vec.length_squared()
                    
                    if line_len_sq == 0: continue

                    t = max(0, min(1, point_vec.dot(line_vec) / line_len_sq))
                    closest_point = p1 + t * line_vec
                    
                    dist_sq = new_pos.distance_squared_to(closest_point)
                    
                    if dist_sq < self.BALL_RADIUS ** 2:
                        # Collision occurred
                        wall_hit_reward += self.REWARD_WALL_HIT
                        normal = self.wall_normals[i]
                        
                        # Reflect velocity
                        ball['vel'] = ball['vel'].reflect(normal)
                        
                        # Move ball out of wall to prevent sticking
                        penetration_depth = self.BALL_RADIUS - new_pos.distance_to(closest_point)
                        new_pos += normal * penetration_depth * 1.1

                        # Spawn force field and particles
                        self._create_force_field(closest_point, ball['color'])
                        self._create_particles(closest_point, ball['color'])
                        # sfx: wall_hit_sound
                        break # Assume one collision per frame
                
                ball['pos'] = new_pos
        return wall_hit_reward

    def _create_force_field(self, pos, color):
        self.force_fields.append({
            'pos': pos,
            'color': color,
            'life': self.FORCE_FIELD_DURATION,
            'radius': 0,
            'max_radius': self.FORCE_FIELD_MAX_RADIUS,
        })

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(50, 150)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.uniform(0.3, self.PARTICLE_LIFESPAN),
                'max_life': self.PARTICLE_LIFESPAN,
                'color': color
            })

    def _update_force_fields(self, dt):
        for ff in self.force_fields:
            ff['life'] -= dt
            # Ease-out quad for radius growth
            progress = 1 - (ff['life'] / self.FORCE_FIELD_DURATION)
            ease_out_progress = 1 - (1 - progress) ** 2
            ff['radius'] = ease_out_progress * ff['max_radius']
        
        self.force_fields = [ff for ff in self.force_fields if ff['life'] > 0]

    def _update_particles(self, dt):
        for p in self.particles:
            p['life'] -= dt
            p['pos'] += p['vel'] * dt
            p['vel'] *= 0.95 # Damping
        
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        return self.time_remaining <= 0 or self.score >= self.WIN_SCORE

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
            "time_remaining": self.time_remaining,
        }

    def _render_game(self):
        self._render_force_fields()
        self._render_walls()
        self._render_particles()
        self._render_balls()
        self._render_aimer()

    def _render_walls(self):
        for p1, p2 in self.walls:
            pygame.draw.line(self.screen, self.COLOR_WALL, p1, p2, self.WALL_THICKNESS)

    def _render_force_fields(self):
        # Use a temporary surface for additive blending
        ff_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for ff in self.force_fields:
            life_ratio = ff['life'] / self.FORCE_FIELD_DURATION
            # Pulsing effect on creation, then fade out
            alpha = int(255 * (1 - (1 - life_ratio)**3)) if life_ratio > 0.5 else int(255 * (life_ratio / 0.5))
            alpha = max(0, min(alpha, 255))
            
            pulse_factor = 1.0 + 0.1 * math.sin( (1-life_ratio) * 10 * math.pi) * life_ratio**2
            radius = int(ff['radius'] * pulse_factor)
            
            color = ff['color']
            
            # Draw filled circle with alpha
            pygame.gfxdraw.filled_circle(ff_surface, int(ff['pos'].x), int(ff['pos'].y), radius, (*color, 30))
            # Draw anti-aliased outline
            pygame.gfxdraw.aacircle(ff_surface, int(ff['pos'].x), int(ff['pos'].y), radius, (*color, alpha))
        
        self.screen.blit(ff_surface, (0, 0))

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(self.BALL_RADIUS * 0.3 * life_ratio)
            if radius > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), radius)

    def _render_balls(self):
        for ball in self.balls:
            pos = (int(ball['pos'].x), int(ball['pos'].y))
            color = ball['color']
            # Draw glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS + 3, (*color, 30))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS + 6, (*color, 15))
            # Draw ball
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, color)

    def _render_aimer(self):
        ready_balls = [b for b in self.balls if b['state'] == 'ready']
        if not ready_balls: return

        aimer_len = 30
        end_x = self.launch_pos[0] + aimer_len * math.cos(self.launch_angle)
        end_y = self.launch_pos[1] + aimer_len * math.sin(self.launch_angle)
        
        pygame.draw.line(self.screen, self.COLOR_AIMER, self.launch_pos, (end_x, end_y), 2)
        pygame.draw.circle(self.screen, self.COLOR_AIMER, (int(end_x), int(end_y)), 4)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_text = self.font.render(f"TIME: {self.time_remaining:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        if self.game_over:
            result_text_str = "VICTORY!" if self.score >= self.WIN_SCORE else "TIME UP"
            result_color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            result_font = pygame.font.SysFont("Consolas", 60, bold=True)
            result_text = result_font.render(result_text_str, True, result_color)
            text_rect = result_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(result_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a graphical display.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bounce Field")
    
    running = True
    total_reward = 0.0

    while running:
        # --- Action mapping for human player ---
        movement = 0 # none
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, just need to display it
        # Need to transpose it back for pygame's blit
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()