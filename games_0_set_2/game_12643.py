import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:12:17.868330
# Source Brief: brief_02643.md
# Brief Index: 2643
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player navigates a procedurally generated tunnel.
    The player must change their shape (circle, square, triangle) to match
    the shape of upcoming gaps in the tunnel walls.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a procedurally generated tunnel by changing your shape (circle, square, triangle) to match "
        "upcoming gaps in the walls."
    )
    user_guide = "Use the arrow keys (↑↓←→) to move and press space to cycle through shapes."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    
    # Colors
    COLOR_BG = (10, 0, 25)
    COLOR_WALL = (40, 40, 60)
    COLOR_GAP_OUTLINE = (255, 255, 255)
    
    SHAPE_CIRCLE, SHAPE_SQUARE, SHAPE_TRIANGLE = 0, 1, 2
    SHAPE_PROPS = {
        SHAPE_CIRCLE: {
            "name": "Circle", "color": (50, 150, 255), "glow_color": (150, 200, 255),
            "speed": 6.0, "maneuverability": 2.0, "size": 18
        },
        SHAPE_SQUARE: {
            "name": "Square", "color": (255, 220, 50), "glow_color": (255, 240, 150),
            "speed": 4.5, "maneuverability": 3.0, "size": 32 # size is side length
        },
        SHAPE_TRIANGLE: {
            "name": "Triangle", "color": (255, 50, 100), "glow_color": (255, 150, 180),
            "speed": 3.0, "maneuverability": 4.0, "size": 40 # size is height
        }
    }

    # Game parameters
    MAX_STEPS = 1000
    WIN_DISTANCE = 5000
    INITIAL_TRANSFORMATIONS = 25
    PLAYER_Y_POS = 320 # Player's fixed Y position on screen
    SEGMENT_HEIGHT = 150
    BASE_GAP_WIDTH = 180

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_x = 0
        self.player_shape = 0
        self.transformations_left = 0
        self.distance_traveled = 0
        self.difficulty_step_counter = 0
        self.gap_width_scale = 1.0
        self.space_was_held = False
        self.tunnel_segments = deque()
        self.particles = deque()
        self.stars = []
        self.game_over_message = ""
        
        # self.reset() is called by the wrapper, but for standalone use we might need it.
        # self.validate_implementation() is for debugging during dev, not needed in final code.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_x = self.WIDTH // 2
        self.player_shape = self.SHAPE_CIRCLE
        self.transformations_left = self.INITIAL_TRANSFORMATIONS
        self.distance_traveled = 0.0
        
        self.difficulty_step_counter = 0
        self.gap_width_scale = 1.0
        
        self.space_was_held = True # Prevent transform on first frame if space is held
        
        self.tunnel_segments.clear()
        self.particles.clear()
        self._generate_stars()
        self._generate_initial_tunnel()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Unpack action and apply player logic ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.01 # Small reward for surviving
        self._update_player(movement, space_held)

        # --- 2. Update game world ---
        current_speed = self.SHAPE_PROPS[self.player_shape]["speed"]
        self.distance_traveled += current_speed
        self._update_tunnel(current_speed)
        self._update_particles()
        self._update_stars(current_speed)
        self._update_difficulty()

        # --- 3. Check for events and termination ---
        reward += self._check_gaps_traversed()
        
        collision = self._check_collisions()
        
        terminated = False
        if collision:
            terminated = True
            reward = -100
            self.game_over_message = "COLLISION!"
        elif self.transformations_left < 0:
            terminated = True
            reward = -50
            self.transformations_left = 0 # Clamp for display
            self.game_over_message = "NO TRANSFORMS LEFT"
        elif self.distance_traveled >= self.WIN_DISTANCE:
            terminated = True
            reward = +100
            self.game_over_message = "TUNNEL COMPLETE!"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over_message = "TIME LIMIT REACHED"

        self.game_over = terminated
        self.score += reward
        self.steps += 1
        
        truncated = False # This environment does not truncate based on time limit
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, movement, space_held):
        # Nudge movement
        maneuverability = self.SHAPE_PROPS[self.player_shape]["maneuverability"]
        if movement == 1: self.PLAYER_Y_POS = max(50, self.PLAYER_Y_POS - maneuverability) # Up
        elif movement == 2: self.PLAYER_Y_POS = min(self.HEIGHT - 50, self.PLAYER_Y_POS + maneuverability) # Down
        elif movement == 3: self.player_x = max(0, self.player_x - maneuverability) # Left
        elif movement == 4: self.player_x = min(self.WIDTH, self.player_x + maneuverability) # Right

        # Shape transformation
        space_pressed = space_held and not self.space_was_held
        if space_pressed and self.transformations_left > 0:
            self.player_shape = (self.player_shape + 1) % 3
            self.transformations_left -= 1
            self._spawn_particles(20, self.SHAPE_PROPS[self.player_shape]["color"])
        self.space_was_held = space_held
        
        # If player tries to transform with 0 left, it's a terminal state
        if space_pressed and self.transformations_left <= 0:
            self.transformations_left = -1 # Signal for termination check


    def _update_tunnel(self, scroll_speed):
        # Scroll existing segments
        for segment in self.tunnel_segments:
            segment['y'] += scroll_speed
        
        # Remove segments that are off-screen
        if self.tunnel_segments and self.tunnel_segments[0]['y'] > self.HEIGHT + self.SEGMENT_HEIGHT:
            self.tunnel_segments.popleft()
            
        # Add new segments if needed
        if self.tunnel_segments:
            last_segment_y = self.tunnel_segments[-1]['y']
            if last_segment_y > 0:
                self._add_tunnel_segment()

    def _update_difficulty(self):
        self.difficulty_step_counter += 1
        if self.difficulty_step_counter > 500:
            self.difficulty_step_counter = 0
            self.gap_width_scale = max(0.6, self.gap_width_scale * 0.95)

    def _check_collisions(self):
        player_screen_y = self.PLAYER_Y_POS
        
        for segment in self.tunnel_segments:
            # Check if player is vertically aligned with this segment's gap
            if segment['y'] < player_screen_y < segment['y'] + 10: # 10 is a small tolerance window
                gap_x = segment['gap_x']
                gap_width = segment['gap_width']
                wall_left = gap_x - gap_width / 2
                wall_right = gap_x + gap_width / 2

                props = self.SHAPE_PROPS[self.player_shape]
                size = props['size']
                
                player_left, player_right = 0, 0
                if self.player_shape == self.SHAPE_CIRCLE:
                    player_left = self.player_x - size
                    player_right = self.player_x + size
                elif self.player_shape == self.SHAPE_SQUARE:
                    player_left = self.player_x - size / 2
                    player_right = self.player_x + size / 2
                elif self.player_shape == self.SHAPE_TRIANGLE:
                    width = size * math.sqrt(3) / 2
                    player_left = self.player_x - width / 2
                    player_right = self.player_x + width / 2

                # Wall collision
                if player_left < wall_left or player_right > wall_right:
                    return True
                
                # Shape mismatch collision
                if self.player_shape != segment['gap_shape']:
                    return True
                    
        return False

    def _check_gaps_traversed(self):
        reward = 0
        player_screen_y = self.PLAYER_Y_POS
        for segment in self.tunnel_segments:
            if not segment['traversed'] and segment['y'] > player_screen_y:
                segment['traversed'] = True
                if self.player_shape == segment['gap_shape']:
                    reward += 1.0
                else:
                    reward -= 1.0 
        return reward

    def _generate_initial_tunnel(self):
        for i in range(-2, (self.HEIGHT // self.SEGMENT_HEIGHT) + 2):
            self._add_tunnel_segment(initial_y=i * self.SEGMENT_HEIGHT)

    def _add_tunnel_segment(self, initial_y=None):
        if self.tunnel_segments:
            last_segment = self.tunnel_segments[-1]
            y = last_segment['y'] - self.SEGMENT_HEIGHT
            
            max_shift = 80
            new_gap_x = last_segment['gap_x'] + self.np_random.uniform(-max_shift, max_shift)
            min_gap_pos = self.BASE_GAP_WIDTH * self.gap_width_scale / 2 + 20
            new_gap_x = np.clip(new_gap_x, min_gap_pos, self.WIDTH - min_gap_pos)
        else:
            y = initial_y if initial_y is not None else -self.SEGMENT_HEIGHT
            new_gap_x = self.WIDTH / 2
            
        new_segment = {
            'y': y,
            'gap_x': new_gap_x,
            'gap_width': self.BASE_GAP_WIDTH * self.gap_width_scale,
            'gap_shape': self.np_random.integers(0, 3),
            'traversed': False
        }
        self.tunnel_segments.append(new_segment)
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_stars()
        self._draw_tunnel()
        self._draw_particles()
        if not self.game_over:
            self._draw_player()

    def _render_ui(self):
        trans_text = self.font_ui.render(f"Transforms: {self.transformations_left}", True, (255, 255, 255))
        self.screen.blit(trans_text, (10, 10))

        dist_text = self.font_ui.render(f"Distance: {int(self.distance_traveled)}", True, (255, 255, 255))
        self.screen.blit(dist_text, (self.WIDTH - dist_text.get_width() - 10, 10))
        
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            over_text = self.font_game_over.render(self.game_over_message, True, (255, 255, 255))
            text_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_text, text_rect)

    def _draw_player(self):
        props = self.SHAPE_PROPS[self.player_shape]
        color = props["color"]
        glow_color = props["glow_color"]
        size = props["size"]
        x, y = int(self.player_x), int(self.PLAYER_Y_POS)

        glow_radius = int(size * 1.8)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*glow_color, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        if self.player_shape == self.SHAPE_CIRCLE:
            pygame.gfxdraw.aacircle(self.screen, x, y, size, color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, size, color)
        elif self.player_shape == self.SHAPE_SQUARE:
            half = size // 2
            rect = pygame.Rect(x - half, y - half, size, size)
            pygame.draw.rect(self.screen, color, rect)
        elif self.player_shape == self.SHAPE_TRIANGLE:
            h = size
            w = h * math.sqrt(3) / 2
            p1 = (x, y - h * 2/3)
            p2 = (x - w / 2, y + h / 3)
            p3 = (x + w / 2, y + h / 3)
            points = [p1, p2, p3]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            
    def _draw_tunnel(self):
        for i in range(len(self.tunnel_segments) - 1):
            s1 = self.tunnel_segments[i]
            s2 = self.tunnel_segments[i+1]
            
            y1, y2 = int(s1['y']), int(s2['y'])
            
            p_left = [
                (0, y1), (s1['gap_x'] - s1['gap_width']/2, y1),
                (s2['gap_x'] - s2['gap_width']/2, y2), (0, y2)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_WALL, p_left)

            p_right = [
                (self.WIDTH, y1), (s1['gap_x'] + s1['gap_width']/2, y1),
                (s2['gap_x'] + s2['gap_width']/2, y2), (self.WIDTH, y2)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_WALL, p_right)

            gap_y = int(s1['y'])
            gap_x = int(s1['gap_x'])
            gap_w = int(s1['gap_width'])
            gap_shape = s1['gap_shape']
            
            self._draw_gap_indicator(gap_x, gap_y, gap_w, gap_shape)

    def _draw_gap_indicator(self, x, y, width, shape):
        props = self.SHAPE_PROPS[shape]
        color = props["glow_color"]
        size = props["size"] * 1.5 
        
        if y < 0 or y > self.HEIGHT: return

        if shape == self.SHAPE_CIRCLE:
            pygame.gfxdraw.aacircle(self.screen, x, y, int(size), color)
        elif shape == self.SHAPE_SQUARE:
            half = int(size / 2)
            pygame.draw.rect(self.screen, color, (x - half, y - half, size, size), 2)
        elif shape == self.SHAPE_TRIANGLE:
            h = size
            w = h * math.sqrt(3) / 2
            points = [(x, y - h * 2/3), (x - w/2, y + h/3), (x + w/2, y + h/3)]
            pygame.draw.aalines(self.screen, color, True, points)

    def _spawn_particles(self, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': [self.player_x, self.PLAYER_Y_POS],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _draw_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            life_ratio = p['life'] / 30.0
            radius = int(life_ratio * 4)
            if radius > 0:
                pygame.draw.circle(self.screen, p['color'], pos, radius)
                
    def _update_particles(self):
        for p in list(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _generate_stars(self):
        self.stars.clear()
        for _ in range(150):
            self.stars.append({
                'x': self.np_random.uniform(0, self.WIDTH),
                'y': self.np_random.uniform(0, self.HEIGHT),
                'depth': self.np_random.uniform(0.1, 0.8)
            })

    def _update_stars(self, scroll_speed):
        for star in self.stars:
            star['y'] += scroll_speed * star['depth']
            if star['y'] > self.HEIGHT:
                star['y'] = 0
                star['x'] = self.np_random.uniform(0, self.WIDTH)

    def _draw_stars(self):
        for star in self.stars:
            brightness = int(200 * star['depth'])
            color = (brightness, brightness, brightness)
            size = int(2 * star['depth'])
            pygame.draw.rect(self.screen, color, (star['x'], star['y'], size, size))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_traveled": self.distance_traveled,
            "transformations_left": self.transformations_left if self.transformations_left >= 0 else 0
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # --- Manual Play Example ---
    # Re-enable display for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Shape Shifter Tunnel")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # --- Action mapping for human play ---
        movement = 0 # none
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Info: {info}")
    env.close()