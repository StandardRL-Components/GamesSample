import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:31:06.939763
# Source Brief: brief_00505.md
# Brief Index: 505
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls mirrors to reflect light beams.
    The goal is to destroy all obstacles by combining beams and then hit a central target.

    **Visuals:**
    - Dark, futuristic grid background.
    - Vibrant, glowing light beams that change color when combined.
    - Geometric shapes for obstacles, ice, and the target.
    - Particle effects for destruction and melting events.
    - Clear UI indicating remaining obstacles and game status.

    **Gameplay:**
    - The agent selects one of several mirrors on the screen.
    - The agent can rotate the selected mirror clockwise or counter-clockwise.
    - Light beams emanate from the screen edges.
    - Beams reflect off mirrors.
    - Beams melt blue 'ice' blocks on contact.
    - Beams must be combined (overlapped) to create a more powerful beam.
    - Powerful beams are required to destroy grey 'obstacles'.
    - The game is won by hitting the central gold target with any beam *after* all obstacles have been destroyed.

    **Action Space `MultiDiscrete([5, 2, 2])`:**
    - `action[0]` (Movement):
        - 0: No-op
        - 1: Rotate selected mirror clockwise.
        - 2: Rotate selected mirror counter-clockwise.
        - 3: Select previous mirror.
        - 4: Select next mirror.
    - `action[1]` (Space): Unused.
    - `action[2]` (Shift): Unused.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Reflect and combine light beams using rotating mirrors. Destroy all obstacles with powerful, "
        "combined beams, then hit the final target to win."
    )
    user_guide = (
        "Controls: Use ←→ to select a mirror and ↑↓ to rotate it. Combine beams to destroy obstacles."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    BEAM_MAX_REFLECTIONS = 12
    OBSTACLE_DESTROY_THRESHOLD = 2 # Needs a combined beam of at least this power

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_OBSTACLE = (120, 120, 140)
    COLOR_ICE = (170, 220, 255)
    COLOR_TARGET = (255, 215, 0)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_SELECTED_MIRROR = (0, 255, 255)
    BEAM_COLORS = {
        1: (255, 50, 50),   # Red
        2: (255, 255, 50),  # Yellow
        3: (50, 255, 50),   # Green
        4: (255, 255, 255)  # White
    }

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)

        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Game state variables are initialized in reset()
        self.mirrors = []
        self.ice_blocks = []
        self.obstacles = []
        self.emitters = []
        self.target = None
        self.particles = []
        self.beam_paths = []
        self.selected_mirror_idx = 0
        self.obstacles_total = 0
        self.obstacles_destroyed_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.selected_mirror_idx = 0
        
        self._setup_level()
        self.obstacles_total = len(self.obstacles)
        self.obstacles_destroyed_count = 0

        # Initial beam calculation
        self.beam_paths, _ = self._calculate_beams_and_interactions()

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Initializes the positions of all game entities for a new episode."""
        self.target = {'pos': pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2), 'radius': 15, 'hit': False}

        self.emitters = [
            {'pos': pygame.Vector2(1, self.SCREEN_HEIGHT * 0.25), 'dir': pygame.Vector2(1, 0)},
            {'pos': pygame.Vector2(1, self.SCREEN_HEIGHT * 0.75), 'dir': pygame.Vector2(1, 0)},
            {'pos': pygame.Vector2(self.SCREEN_WIDTH - 1, self.SCREEN_HEIGHT * 0.25), 'dir': pygame.Vector2(-1, 0)},
            {'pos': pygame.Vector2(self.SCREEN_WIDTH - 1, self.SCREEN_HEIGHT * 0.75), 'dir': pygame.Vector2(-1, 0)},
        ]

        self.mirrors = [
            {'pos': pygame.Vector2(150, 100), 'angle': 45.0, 'size': 40},
            {'pos': pygame.Vector2(490, 100), 'angle': -45.0, 'size': 40},
            {'pos': pygame.Vector2(150, 300), 'angle': -45.0, 'size': 40},
            {'pos': pygame.Vector2(490, 300), 'angle': 45.0, 'size': 40},
            {'pos': pygame.Vector2(320, 200), 'angle': 90.0, 'size': 50},
        ]

        self.ice_blocks = [
            {'rect': pygame.Rect(220, 50, 20, 20), 'active': True},
            {'rect': pygame.Rect(400, 330, 20, 20), 'active': True},
        ]

        self.obstacles = [
            {'rect': pygame.Rect(310, 80, 20, 40), 'active': True},
            {'rect': pygame.Rect(310, 280, 20, 40), 'active': True},
        ]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # 1. Handle player action
        movement = action[0]
        self._handle_input(movement)

        # 2. Update game state (beams and interactions)
        self.beam_paths, interactions = self._calculate_beams_and_interactions()
        
        # 3. Apply interactions and calculate rewards
        reward += self._apply_interactions(interactions)
        self.score += reward

        # 4. Update particles
        self._update_particles()

        # 5. Check for termination conditions
        terminated = False
        all_obstacles_destroyed = self.obstacles_destroyed_count == self.obstacles_total
        
        if self.target['hit'] and all_obstacles_destroyed:
            # # sound: win_sound
            reward += 5.0 # Reward for final target hit
            reward += 100.0 # Huge reward for winning
            self.score += 105.0
            self.game_over = True
            terminated = True
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            truncated = True
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement_action):
        """Updates mirror state based on agent's action."""
        if not self.mirrors: return

        angle_step = 5.0
        if movement_action == 1:  # Rotate clockwise
            self.mirrors[self.selected_mirror_idx]['angle'] += angle_step
        elif movement_action == 2:  # Rotate counter-clockwise
            self.mirrors[self.selected_mirror_idx]['angle'] -= angle_step
        elif movement_action == 3:  # Select previous
            self.selected_mirror_idx = (self.selected_mirror_idx - 1) % len(self.mirrors)
        elif movement_action == 4:  # Select next
            self.selected_mirror_idx = (self.selected_mirror_idx + 1) % len(self.mirrors)
        
        if movement_action in [1, 2]:
            self.mirrors[self.selected_mirror_idx]['angle'] %= 360

    def _apply_interactions(self, interactions):
        """Applies game logic changes from beam interactions and returns rewards."""
        reward = 0.0
        
        # Melt Ice
        for i in interactions['ice_hit']:
            if self.ice_blocks[i]['active']:
                self.ice_blocks[i]['active'] = False
                reward += 0.1
                # # sound: ice_melt
                self._spawn_particles(self.ice_blocks[i]['rect'].center, self.COLOR_ICE)

        # Destroy Obstacles
        for i in interactions['obstacles_hit']:
            if self.obstacles[i]['active']:
                self.obstacles[i]['active'] = False
                self.obstacles_destroyed_count += 1
                reward += 0.5
                # # sound: obstacle_destroy
                self._spawn_particles(self.obstacles[i]['rect'].center, self.COLOR_OBSTACLE, 30)
        
        self.target['hit'] = interactions['target_hit']
        return reward

    def _calculate_beams_and_interactions(self):
        """Traces all beams and detects collisions."""
        beam_paths = []
        
        # This grid tracks beam power at different locations
        power_grid = np.zeros((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), dtype=int)
        
        for emitter in self.emitters:
            path, grid_updates = self._trace_beam(emitter['pos'], emitter['dir'])
            beam_paths.append(path)
            for x, y in grid_updates:
                if 0 <= x < self.SCREEN_WIDTH and 0 <= y < self.SCREEN_HEIGHT:
                    power_grid[x, y] += 1

        # Check interactions based on the final power grid
        interactions = {'ice_hit': set(), 'obstacles_hit': set(), 'target_hit': False}
        
        for i, ice in enumerate(self.ice_blocks):
            if ice['active'] and self._is_rect_hit(ice['rect'], power_grid, 1):
                interactions['ice_hit'].add(i)

        for i, obs in enumerate(self.obstacles):
            if obs['active'] and self._is_rect_hit(obs['rect'], power_grid, self.OBSTACLE_DESTROY_THRESHOLD):
                interactions['obstacles_hit'].add(i)

        if self._is_target_hit(self.target, power_grid, 1):
            interactions['target_hit'] = True

        return beam_paths, interactions

    def _is_rect_hit(self, rect, grid, threshold):
        """Checks if a rectangle is hit by sufficient beam power."""
        x, y, w, h = rect
        sub_grid = grid[max(0, x):min(self.SCREEN_WIDTH, x+w), max(0, y):min(self.SCREEN_HEIGHT, y+h)]
        return np.any(sub_grid >= threshold)
    
    def _is_target_hit(self, target, grid, threshold):
        """Checks if the circular target is hit."""
        x, y = int(target['pos'].x), int(target['pos'].y)
        r = int(target['radius'])
        for i in range(max(0, x-r), min(self.SCREEN_WIDTH, x+r)):
            for j in range(max(0, y-r), min(self.SCREEN_HEIGHT, y+r)):
                if (i-x)**2 + (j-y)**2 <= r**2:
                    if grid[i, j] >= threshold:
                        return True
        return False

    def _trace_beam(self, start_pos, start_dir):
        """Follow a single beam's path, calculating reflections."""
        path_segments = []
        grid_updates = set()
        
        current_pos = pygame.Vector2(start_pos)
        current_dir = pygame.Vector2(start_dir).normalize()

        for _ in range(self.BEAM_MAX_REFLECTIONS):
            intersections = []

            # Check for mirror intersections
            for i, mirror in enumerate(self.mirrors):
                p1, p2 = self._get_mirror_endpoints(mirror)
                intersect_pt = self._get_line_segment_intersection(current_pos, current_dir, p1, p2)
                if intersect_pt:
                    dist = current_pos.distance_to(intersect_pt)
                    intersections.append({'dist': dist, 'pt': intersect_pt, 'type': 'mirror', 'idx': i})
            
            # Check for screen boundary intersections
            t_x = float('inf')
            if current_dir.x > 0: t_x = (self.SCREEN_WIDTH - current_pos.x) / current_dir.x
            elif current_dir.x < 0: t_x = -current_pos.x / current_dir.x
            
            t_y = float('inf')
            if current_dir.y > 0: t_y = (self.SCREEN_HEIGHT - current_pos.y) / current_dir.y
            elif current_dir.y < 0: t_y = -current_pos.y / current_dir.y
            
            t = min(t_x, t_y)
            if t > 1e-6:
                end_pt = current_pos + t * current_dir
                intersections.append({'dist': t, 'pt': end_pt, 'type': 'boundary'})

            if not intersections: break
            
            # Find the closest intersection
            closest = min(intersections, key=lambda x: x['dist'])
            
            # Add segment to path
            segment_start = pygame.Vector2(current_pos)
            segment_end = pygame.Vector2(closest['pt'])
            path_segments.append((segment_start, segment_end))
            
            # Add points along the segment to the grid update set
            num_steps = int(segment_start.distance_to(segment_end))
            for i in range(num_steps):
                p = segment_start.lerp(segment_end, i / max(1, num_steps-1))
                grid_updates.add((int(p.x), int(p.y)))

            current_pos = closest['pt']
            
            if closest['type'] == 'boundary':
                break
            
            if closest['type'] == 'mirror':
                # # sound: beam_reflect
                mirror = self.mirrors[closest['idx']]
                mirror_angle = math.radians(mirror['angle'])
                normal = pygame.Vector2(math.sin(mirror_angle), -math.cos(mirror_angle))
                current_dir = current_dir.reflect(normal).normalize()
                current_pos += current_dir * 1e-4 # Epsilon push
        
        return path_segments, grid_updates

    def _get_mirror_endpoints(self, mirror):
        """Calculate the start and end points of a mirror line segment."""
        pos = mirror['pos']
        angle = math.radians(mirror['angle'])
        half_size = mirror['size'] / 2
        dx = half_size * math.cos(angle)
        dy = half_size * math.sin(angle)
        return (pos - pygame.Vector2(dx, dy), pos + pygame.Vector2(dx, dy))

    def _get_line_segment_intersection(self, p1, d1, p2, p3):
        """Find intersection of a ray (p1, d1) and a line segment (p2, p3)."""
        d2 = p3 - p2
        cross = d1.x * d2.y - d1.y * d2.x
        if abs(cross) < 1e-8: return None # Parallel lines

        t = ((p2.x - p1.x) * d2.y - (p2.y - p1.y) * d2.x) / cross
        u = ((p2.x - p1.x) * d1.y - (p2.y - p1.y) * d1.x) / cross

        if 1e-6 < t and 0 <= u <= 1:
            return p1 + t * d1
        return None

    def _spawn_particles(self, pos, color, count=20, max_speed=2):
        """Create a burst of particles."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifespan': random.randint(20, 40),
                'color': color,
                'radius': random.uniform(1, 3)
            })

    def _update_particles(self):
        """Move particles and decrease their lifespan."""
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['vel'] *= 0.95 # Damping

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "obstacles_remaining": self.obstacles_total - self.obstacles_destroyed_count,
            "selected_mirror": self.selected_mirror_idx
        }

    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

    def _render_game(self):
        # Render beams first
        self._render_beams()
        
        # Render game entities
        if self.target:
            self._render_glow_circle(self.screen, self.target['pos'], self.target['radius'], self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, int(self.target['pos'].x), int(self.target['pos'].y), int(self.target['radius']), self.COLOR_TARGET)

        for ice in self.ice_blocks:
            if ice['active']:
                pygame.draw.rect(self.screen, self.COLOR_ICE, ice['rect'])
                pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_ICE), ice['rect'], 2)

        for obs in self.obstacles:
            if obs['active']:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])
                pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_OBSTACLE), obs['rect'], 2)

        for i, mirror in enumerate(self.mirrors):
            p1, p2 = self._get_mirror_endpoints(mirror)
            color = self.COLOR_SELECTED_MIRROR if i == self.selected_mirror_idx else (200, 200, 200)
            pygame.draw.line(self.screen, color, p1, p2, 4)

        for emitter in self.emitters:
            pygame.draw.circle(self.screen, self.BEAM_COLORS[1], emitter['pos'], 5)

        # Render particles on top
        self._render_particles()

    def _render_beams(self):
        """Renders all beam paths with glowing effects."""
        power_grid = np.zeros((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), dtype=int)
        
        # First pass: populate the power grid
        for path in self.beam_paths:
            for start, end in path:
                num_steps = int(start.distance_to(end))
                for i in range(num_steps):
                    p = start.lerp(end, i / max(1, num_steps-1))
                    x, y = int(p.x), int(p.y)
                    if 0 <= x < self.SCREEN_WIDTH and 0 <= y < self.SCREEN_HEIGHT:
                        power_grid[x, y] += 1

        # Second pass: draw the beams with color based on power
        for path in self.beam_paths:
            for start, end in path:
                mid_point = start.lerp(end, 0.5)
                mx, my = int(mid_point.x), int(mid_point.y)
                if 0 <= mx < self.SCREEN_WIDTH and 0 <= my < self.SCREEN_HEIGHT:
                    power = power_grid[mx, my]
                    color = self.BEAM_COLORS.get(power, self.BEAM_COLORS[4])
                    
                    # Glow effect
                    pygame.draw.line(self.screen, (*color, 30), start, end, 7)
                    pygame.draw.line(self.screen, (*color, 50), start, end, 5)
                    # Core beam
                    pygame.draw.aaline(self.screen, color, start, end, 2)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40.0))
            color = (*p['color'], alpha)
            # Using simple circles for performance
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

    def _render_ui(self):
        obs_text = f"OBSTACLES: {self.obstacles_total - self.obstacles_destroyed_count}/{self.obstacles_total}"
        text_surf = self.font_ui.render(obs_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.target['hit'] and (self.obstacles_destroyed_count == self.obstacles_total) else "TIME UP"
            win_condition = self.target['hit'] and (self.obstacles_destroyed_count == self.obstacles_total)
            msg_surf = self.font_msg.render(msg, True, self.COLOR_TARGET if win_condition else self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def _render_glow_circle(self, surface, center, radius, color):
        """Renders a circle with a glowing aura."""
        for i in range(4, 0, -1):
            alpha = 60 - i * 12
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, int(center.x), int(center.y), int(radius + i * 2), glow_color)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    if os.environ.get("SDL_VIDEODRIVER", "") == "dummy":
        print("Cannot run manual test in headless mode. Skipping.")
    else:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Light Bender")
        clock = pygame.time.Clock()
        
        done = False
        total_reward = 0
        
        print("\n--- Manual Control ---")
        print(GameEnv.user_guide)
        print("Q: Quit")
        
        action = [0, 0, 0] # [movement, space, shift]
        
        while not done:
            # Handle user input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        done = True
                    # Map keys to actions based on user_guide
                    if event.key == pygame.K_UP: action[0] = 1 # Rotate clockwise
                    elif event.key == pygame.K_DOWN: action[0] = 2 # Rotate counter-clockwise
                    elif event.key == pygame.K_LEFT: action[0] = 3 # Select previous
                    elif event.key == pygame.K_RIGHT: action[0] = 4 # Select next
                    
            # For auto-advancing games, we always step.
            # For non-auto-advancing, you might only step on key presses.
            # Since auto_advance is True, we step every frame.
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            action = [0, 0, 0] # Reset action after one step
            
            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward}")
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

            # Render the observation to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit to 30 FPS
            
        env.close()