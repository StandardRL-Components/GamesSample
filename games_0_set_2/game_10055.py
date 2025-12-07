import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:50:44.060066
# Source Brief: brief_00055.md
# Brief Index: 55
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for effects."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = random.randint(20, 40)
        self.radius = random.randint(3, 6)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius -= 0.1

    def draw(self, surface):
        if self.lifespan > 0 and self.radius > 0:
            alpha = int(255 * (self.lifespan / 40))
            color = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a beam of light through a maze of obstacles. Hit colored prisms to gain power-ups and reach the final target."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to change the direction of the light beam."
    )
    auto_advance = True

    # --- Colors and Constants ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (15, 18, 33)
    COLOR_OBSTACLE = (50, 50, 60)
    COLOR_OBSTACLE_GLOW = (80, 80, 90)
    COLOR_TARGET = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_RED = (255, 50, 50)
    COLOR_GREEN = (50, 255, 50)
    COLOR_BLUE = (50, 100, 255)
    PRISM_COLORS = {"red": COLOR_RED, "green": COLOR_GREEN, "blue": COLOR_BLUE}

    MAX_STEPS = 1000
    MAX_ATTEMPTS = 3
    BEAM_SPEED = 5
    
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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_chain = pygame.font.SysFont("Consolas", 24, bold=True)

        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.attempts_left = 0
        self.beam_path = []
        self.beam_head = (0,0)
        self.beam_direction = (0,0)
        self.beam_color = (0,0,0)
        self.beam_width = 0
        self.active_effects = {}
        self.prisms = []
        self.obstacles = []
        self.target = None
        self.particles = []
        self.last_distance_to_target = 0
        self.last_prism_color = None
        self.chain_length = 0
        self.stars = []

        self._generate_stars()
        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.attempts_left = self.MAX_ATTEMPTS
        self._reset_level()
        
        return self._get_observation(), self._get_info()

    def _reset_level(self):
        """Resets the level for a new attempt."""
        self.beam_start_pos = (50, self.HEIGHT // 2)
        self.beam_head = self.beam_start_pos
        self.beam_direction = (1, 0) # Start moving right
        self.beam_color = self.COLOR_TARGET
        self.beam_width = 3
        self.beam_path = [(self.beam_head, self.beam_color, self.beam_width)]
        self.active_effects = {'speed': 0, 'width': 0, 'phase': 0}
        self.particles.clear()
        
        self.last_prism_color = None
        self.chain_length = 0
        
        self._generate_layout()
        if self.target:
            self.last_distance_to_target = self._distance(self.beam_head, self.target.center)
        else:
            self.last_distance_to_target = 0

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self._update_beam_direction(movement)
        
        reward = 0
        
        # --- Update game state ---
        self.steps += 1
        self._update_effects()
        self._update_beam()
        self._update_particles()
        
        # --- Rewards and Collisions ---
        # Distance reward
        if self.target:
            current_distance = self._distance(self.beam_head, self.target.center)
            if current_distance < self.last_distance_to_target:
                reward += 0.01
            else:
                reward -= 0.01
            self.last_distance_to_target = current_distance

        # Collision checks
        collision_reward, terminated, attempt_failed = self._handle_collisions()
        reward += collision_reward
        self.score += collision_reward # Score only increases with positive events

        if attempt_failed:
            self.attempts_left -= 1
            if self.attempts_left <= 0:
                reward -= 100 # Final penalty for losing
                self.score -= 100
                terminated = True
            else:
                self._reset_level()
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        
        truncated = False # Truncated is handled by env wrappers
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_beam_direction(self, movement):
        if movement == 1 and self.beam_direction != (0, 1):  # Up
            self.beam_direction = (0, -1)
        elif movement == 2 and self.beam_direction != (0, -1):  # Down
            self.beam_direction = (0, 1)
        elif movement == 3 and self.beam_direction != (1, 0):  # Left
            self.beam_direction = (-1, 0)
        elif movement == 4 and self.beam_direction != (-1, 0):  # Right
            self.beam_direction = (1, 0)
        else: # No-op or reversing
            return # Keep current direction
        
        # Add a new segment to the path when direction changes
        self.beam_path.append((self.beam_head, self.beam_color, self.beam_width))
        
    def _update_beam(self):
        speed_modifier = 2 if self.active_effects['speed'] > 0 else 1
        dx, dy = self.beam_direction
        self.beam_head = (self.beam_head[0] + dx * self.BEAM_SPEED * speed_modifier,
                          self.beam_head[1] + dy * self.BEAM_SPEED * speed_modifier)
        self.beam_path.append((self.beam_head, self.beam_color, self.beam_width))

    def _handle_collisions(self):
        reward, terminated, attempt_failed = 0, False, False
        beam_head_rect = pygame.Rect(self.beam_head[0]-1, self.beam_head[1]-1, 2, 2)

        # Target
        if self.target and beam_head_rect.colliderect(self.target):
            reward += 100
            terminated = True
            self._create_particle_burst(self.beam_head, self.COLOR_TARGET, 50)
            return reward, terminated, attempt_failed

        # Walls
        if not (0 <= self.beam_head[0] <= self.WIDTH and 0 <= self.beam_head[1] <= self.HEIGHT):
            attempt_failed = True
            return reward, terminated, attempt_failed

        # Prisms
        for prism in self.prisms[:]:
            if self._point_in_triangle(self.beam_head, prism['points']):
                reward += 1
                self._create_particle_burst(self.beam_head, self.PRISM_COLORS[prism['type']], 30)
                self.beam_color = self.PRISM_COLORS[prism['type']]
                self.beam_path.append((self.beam_head, self.beam_color, self.beam_width))

                if self.last_prism_color == prism['type']:
                    self.chain_length += 1
                else:
                    self.last_prism_color = prism['type']
                    self.chain_length = 1
                
                if self.chain_length >= 5:
                    reward += 5 # Chain bonus

                if prism['type'] == 'red': self.active_effects['speed'] = 5 * 5
                if prism['type'] == 'green': self.active_effects['width'] = 5 * 5
                if prism['type'] == 'blue': self.active_effects['phase'] = 3 * 5
                
                self.prisms.remove(prism)
                break

        # Obstacles
        can_phase = self.active_effects['phase'] > 0
        if not can_phase:
            for obstacle in self.obstacles:
                if obstacle.colliderect(beam_head_rect):
                    attempt_failed = True
                    break
        
        return reward, terminated, attempt_failed

    def _update_effects(self):
        for effect in self.active_effects:
            if self.active_effects[effect] > 0:
                self.active_effects[effect] -= 1
        
        self.beam_width = 6 if self.active_effects['width'] > 0 else 3

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "attempts": self.attempts_left}

    def _render_game(self):
        self._render_background()
        self._render_obstacles()
        self._render_prisms()
        self._render_target()
        self._render_beam()
        self._render_particles()

    def _render_background(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], (int(star['x']), int(star['y'])), star['radius'])

    def _render_obstacles(self):
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, obstacle, 1)

    def _render_prisms(self):
        for prism in self.prisms:
            self._draw_glowing_triangle(self.screen, self.PRISM_COLORS[prism['type']], prism['points'], 1.5)

    def _render_target(self):
        if self.target:
            self._draw_glowing_circle(self.screen, self.COLOR_TARGET, self.target.center, self.target.width // 2, 2.0)

    def _render_beam(self):
        if len(self.beam_path) < 2: return
        
        path_segments = []
        current_segment = [self.beam_path[0][0]]
        current_color = self.beam_path[0][1]
        current_width = self.beam_path[0][2]

        for i in range(1, len(self.beam_path)):
            pos, color, width = self.beam_path[i]
            prev_pos, prev_color, prev_width = self.beam_path[i-1]
            if color != current_color or width != current_width:
                path_segments.append({'points': current_segment, 'color': current_color, 'width': current_width})
                current_segment = [prev_pos, pos]
                current_color = color
                current_width = width
            else:
                current_segment.append(pos)
        path_segments.append({'points': current_segment, 'color': current_color, 'width': current_width})

        for seg in path_segments:
            if len(seg['points']) > 1:
                self._draw_glowing_lines(self.screen, seg['color'], seg['points'], seg['width'], 1.8)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Attempts
        attempts_text = self.font_ui.render(f"ATTEMPTS: {self.attempts_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(attempts_text, (10, 10))

        # Beam Color
        color_text = self.font_ui.render("BEAM:", True, self.COLOR_UI_TEXT)
        self.screen.blit(color_text, (self.WIDTH - 120, 10))
        self._draw_glowing_circle(self.screen, self.beam_color, (self.WIDTH - 40, 20), 10, 1.5)

        # Chain Length
        if self.chain_length > 1 and self.last_prism_color:
            chain_text = self.font_chain.render(f"CHAIN x{self.chain_length}", True, self.PRISM_COLORS[self.last_prism_color])
            text_rect = chain_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 30))
            self.screen.blit(chain_text, text_rect)
            
    # --- Generation and Utility ---
    def _generate_layout(self):
        self.obstacles.clear()
        self.prisms.clear()
        
        # Target
        self.target = pygame.Rect(self.WIDTH - 80, self.HEIGHT // 2 - 20, 40, 40)
        
        # Obstacles
        for _ in range(self.np_random.integers(5, 8)):
            w, h = self.np_random.integers(20, 100, size=2)
            x = self.np_random.integers(150, self.WIDTH - w - 100)
            y = self.np_random.integers(0, self.HEIGHT - h)
            new_obstacle = pygame.Rect(x, y, w, h)
            if not new_obstacle.colliderect(self.target):
                self.obstacles.append(new_obstacle)

        # Prisms
        prism_types = list(self.PRISM_COLORS.keys())
        for _ in range(self.np_random.integers(4, 7)):
            ptype = self.np_random.choice(prism_types)
            size = 20
            x = self.np_random.integers(150, self.WIDTH - size - 100)
            y = self.np_random.integers(size, self.HEIGHT - size)
            
            p1 = (x, y)
            p2 = (x + size, y)
            p3 = (x + size / 2, y - size)
            
            new_prism = {'type': ptype, 'points': (p1, p2, p3)}
            
            prism_rect = pygame.Rect(x, y - size, size, size)
            if not any(prism_rect.colliderect(obs) for obs in self.obstacles) and not prism_rect.colliderect(self.target):
                 self.prisms.append(new_prism)

    def _generate_stars(self):
        self.stars = []
        for _ in range(100):
            self.stars.append({
                'x': random.randint(0, self.WIDTH),
                'y': random.randint(0, self.HEIGHT),
                'radius': random.randint(1, 2),
                'color': random.choice([(100, 100, 120), (120, 120, 150), (80, 80, 100)])
            })

    def _create_particle_burst(self, pos, color, count):
        for _ in range(count):
            self.particles.append(Particle(pos[0], pos[1], color))

    @staticmethod
    def _distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @staticmethod
    def _point_in_triangle(p, tri_points):
        p1, p2, p3 = tri_points
        s = p1[1] * p3[0] - p1[0] * p3[1] + (p3[1] - p1[1]) * p[0] + (p1[0] - p3[0]) * p[1]
        t = p1[0] * p2[1] - p1[1] * p2[0] + (p1[1] - p2[1]) * p[0] + (p2[0] - p1[0]) * p[1]

        if (s < 0) != (t < 0) and s != 0 and t != 0:
            return False

        A = -p2[1] * p3[0] + p1[1] * (p3[0] - p2[0]) + p1[0] * (p2[1] - p3[1]) + p2[0] * p3[1]
        if A < 0:
            s = -s
            t = -t
            A = -A
        return s > 0 and t > 0 and (s + t) <= A

    @staticmethod
    def _draw_glowing_circle(surface, color, center, radius, glow_factor):
        glow_radius = int(radius * glow_factor)
        for i in range(glow_radius - radius, 0, -1):
            alpha = int(80 * (1 - i / (glow_radius - radius))) if (glow_radius - radius) > 0 else 80
            glow_color = color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), radius + i, glow_color)
        pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), radius, color)
        pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), radius, color)
        
    @staticmethod
    def _draw_glowing_triangle(surface, color, points, glow_factor):
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)
        glow_color = color + (60,)
        pygame.gfxdraw.filled_polygon(surface, points, glow_color)

    @staticmethod
    def _draw_glowing_lines(surface, color, points, width, glow_factor):
        glow_width = int(width * glow_factor)
        pygame.draw.lines(surface, color, False, points, glow_width)
        pygame.draw.lines(surface, (255,255,255), False, points, width)

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    # Set a real video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Light Beam Maze")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # Default action
    action = np.array([0, 0, 0]) # [movement, space, shift]

    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("Q: Quit")
    
    while not done:
        # Action is reset to no-op unless a key is pressed
        current_action = np.array([0, 0, 0])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Environment Reset ---")
                
                # Update movement part of the action
                if event.key == pygame.K_UP: current_action[0] = 1
                elif event.key == pygame.K_DOWN: current_action[0] = 2
                elif event.key == pygame.K_LEFT: current_action[0] = 3
                elif event.key == pygame.K_RIGHT: current_action[0] = 4
        
        # The game auto-advances, so we always step.
        # If a key was pressed, we use that action, otherwise a no-op.
        obs, reward, terminated, truncated, info = env.step(current_action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()