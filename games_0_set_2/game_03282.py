import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in a headless mode, which is required for server-side execution.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place a crystal. Shift to cycle crystal color."
    )

    game_description = (
        "Strategically place crystals in an isometric cavern to redirect falling orbs into a collection zone. Collect 100 orbs to win."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID_LINE = (35, 45, 65)
    COLOR_WALL = (50, 60, 80)
    COLOR_COLLECTOR = (255, 255, 255)
    COLOR_SPAWNER = (100, 110, 130)
    COLOR_CURSOR = (255, 255, 100)
    
    ORB_COLORS = {
        "red": (255, 80, 80),
        "blue": (80, 150, 255),
        "green": (80, 255, 150),
    }
    
    # Game parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE_X, GRID_SIZE_Y = 16, 12
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 20, 10
    
    MAX_STEPS = 3000
    WIN_SCORE = 100
    MAX_ESCAPED = 5

    # Physics
    GRAVITY = -0.05
    ORB_RADIUS = 5
    CRYSTAL_ATTRACTION_FORCE = 0.15
    WALL_BOUNCE_DAMPING = 0.7

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
        self.font_small = pygame.font.SysFont("sans", 18)
        self.font_large = pygame.font.SysFont("sans", 48)

        # Isometric projection origin
        self.iso_origin_x = self.SCREEN_WIDTH / 2
        self.iso_origin_y = 80
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.escaped_orbs = 0
        self.game_over = False
        self.win = False
        self.reward = 0

        self.cursor_pos = None
        self.orbs = []
        self.crystals = []
        self.particles = []
        
        self.orb_spawn_timer = 0.0
        self.base_orb_spawn_rate = 1.0 # Orbs per second
        self.current_orb_spawn_rate = self.base_orb_spawn_rate

        self.last_space_held = False
        self.last_shift_held = False
        
        self.available_crystal_colors = []
        self.selected_crystal_color_idx = 0
        
        self.np_random = None

    def _iso_to_screen(self, x, y, z=0):
        screen_x = self.iso_origin_x + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.iso_origin_y + (x + y) * self.TILE_HEIGHT_HALF - z
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             self.np_random = np.random.default_rng(seed=seed)
        else:
             self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.escaped_orbs = 0
        self.game_over = False
        self.win = False

        self.cursor_pos = pygame.Vector2(self.GRID_SIZE_X // 2, self.GRID_SIZE_Y // 2)
        
        self.orbs = []
        self.crystals = []
        self.particles = []

        self.current_orb_spawn_rate = self.base_orb_spawn_rate
        self.orb_spawn_timer = 1.0 / self.current_orb_spawn_rate

        self.last_space_held = True # Prevent placing crystal on first frame
        self.last_shift_held = True # Prevent cycling color on first frame

        self.available_crystal_colors = ["red"]
        self.selected_crystal_color_idx = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward = 0
        
        self._handle_input(action)
        
        if not self.game_over:
            self._update_game_logic()
            self._update_difficulty()
        
        self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not truncated: # Game ended by win/loss condition
            if self.win:
                self.reward += 100
            else:
                self.reward -= 100
        
        # Ensure game over state is consistent
        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            self.reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Movement ---
        if movement == 1: self.cursor_pos.y -= 1
        elif movement == 2: self.cursor_pos.y += 1
        elif movement == 3: self.cursor_pos.x -= 1
        elif movement == 4: self.cursor_pos.x += 1
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.GRID_SIZE_X - 1)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.GRID_SIZE_Y - 1)

        # --- Place Crystal (on press) ---
        if space_held and not self.last_space_held:
            self._place_crystal()
        self.last_space_held = space_held

        # --- Cycle Color (on press) ---
        if shift_held and not self.last_shift_held:
            self.selected_crystal_color_idx = (self.selected_crystal_color_idx + 1) % len(self.available_crystal_colors)
        self.last_shift_held = shift_held

    def _place_crystal(self):
        # FIX: pygame.Vector2 does not have a .copy() method.
        # Create a new vector by passing the old one to the constructor.
        pos = pygame.Vector2(self.cursor_pos)
        
        # Check if a crystal is already at this position
        is_occupied = any(c['pos'] == pos for c in self.crystals)
        
        if not is_occupied:
            color_name = self.available_crystal_colors[self.selected_crystal_color_idx]
            self.crystals.append({"pos": pos, "color": color_name})

    def _update_game_logic(self):
        # --- Update Orbs ---
        self._spawn_orbs()
        self._update_orbs()

        # --- Update Particles ---
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _spawn_orbs(self):
        self.orb_spawn_timer -= 1 / 30.0 # Assuming 30 FPS
        if self.orb_spawn_timer <= 0:
            self.orb_spawn_timer += 1.0 / self.current_orb_spawn_rate
            
            spawn_x = self.np_random.uniform(self.GRID_SIZE_X * 0.4, self.GRID_SIZE_X * 0.6)
            spawn_y = self.np_random.uniform(self.GRID_SIZE_Y * 0.1, self.GRID_SIZE_Y * 0.3)
            
            orb_color_options = list(self.ORB_COLORS.keys())
            if self.score < 60: orb_color_options = [c for c in orb_color_options if c != 'green']
            if self.score < 30: orb_color_options = [c for c in orb_color_options if c != 'blue']

            color = self.np_random.choice(orb_color_options)
            
            self.orbs.append({
                "pos": pygame.Vector3(spawn_x, spawn_y, 200),
                "vel": pygame.Vector3(0, 0, 0),
                "color": color,
            })

    def _update_orbs(self):
        collector_pos_2d = pygame.Vector2(self.GRID_SIZE_X / 2, self.GRID_SIZE_Y - 2)

        for orb in self.orbs[:]:
            # Apply gravity
            orb['vel'].z += self.GRAVITY

            # Apply crystal attraction
            attracted = False
            for crystal in self.crystals:
                if orb['color'] == crystal['color']:
                    vec_to_crystal = crystal['pos'] - pygame.Vector2(orb['pos'].x, orb['pos'].y)
                    dist = vec_to_crystal.length()
                    if dist > 0.5:
                        force = self.CRYSTAL_ATTRACTION_FORCE / max(1, dist)
                        accel = vec_to_crystal.normalize() * force
                        orb['vel'].x += accel.x
                        orb['vel'].y += accel.y
                        attracted = True
            
            if attracted:
                self.reward += 0.1
            else:
                # Penalty for not being redirected towards the collection zone
                vec_to_collector = collector_pos_2d - pygame.Vector2(orb['pos'].x, orb['pos'].y)
                if vec_to_collector.dot(pygame.Vector2(orb['vel'].x, orb['vel'].y)) < 0:
                     self.reward -= 0.01

            # Update position
            orb['pos'] += orb['vel']

            # Wall bouncing
            if not (0 < orb['pos'].x < self.GRID_SIZE_X - 1):
                orb['vel'].x *= -self.WALL_BOUNCE_DAMPING
                orb['pos'].x = np.clip(orb['pos'].x, 0, self.GRID_SIZE_X - 1)
            if not (0 < orb['pos'].y < self.GRID_SIZE_Y - 1):
                orb['vel'].y *= -self.WALL_BOUNCE_DAMPING
                orb['pos'].y = np.clip(orb['pos'].y, 0, self.GRID_SIZE_Y - 1)

            # Check for collection or escape
            if orb['pos'].z <= 0:
                dist_to_collector = (pygame.Vector2(orb['pos'].x, orb['pos'].y) - collector_pos_2d).length()
                if dist_to_collector < 2.0:
                    self.score += 1
                    self.reward += 1
                    self._create_particles(orb['pos'], self.ORB_COLORS[orb['color']])
                else:
                    self.escaped_orbs += 1
                    self.reward -= 1
                self.orbs.remove(orb)

    def _update_difficulty(self):
        # Increase spawn rate
        self.current_orb_spawn_rate = self.base_orb_spawn_rate + (self.score // 20) * 0.2
        
        # Unlock new colors
        if self.score >= 30 and "blue" not in self.available_crystal_colors:
            self.available_crystal_colors.append("blue")
        if self.score >= 60 and "green" not in self.available_crystal_colors:
            self.available_crystal_colors.append("green")

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector3(math.cos(angle) * speed, math.sin(angle) * speed, self.np_random.uniform(1, 4))
            lifespan = self.np_random.integers(15, 30)
            # FIX: pygame.Vector3 does not have a .copy() method.
            # Create a new vector by passing the old one to the constructor.
            self.particles.append({"pos": pygame.Vector3(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.win = True
            return True
        if self.escaped_orbs >= self.MAX_ESCAPED:
            self.win = False
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid floor and walls
        self._render_cavern()

        # Draw crystals
        for crystal in self.crystals:
            self._render_crystal(crystal)

        # Draw orbs and particles, sorted by Y-axis for correct layering
        render_list = []
        for orb in self.orbs:
            sx, sy = self._iso_to_screen(orb['pos'].x, orb['pos'].y, orb['pos'].z)
            render_list.append(('orb', sy, orb, sx, sy))
        
        for particle in self.particles:
            sx, sy = self._iso_to_screen(particle['pos'].x, particle['pos'].y, particle['pos'].z)
            render_list.append(('particle', sy, particle, sx, sy))
        
        render_list.sort(key=lambda item: item[1])

        for item in render_list:
            if item[0] == 'orb':
                self._render_orb(item[2], item[3], item[4])
            elif item[0] == 'particle':
                self._render_particle(item[2], item[3], item[4])

        # Draw cursor
        self._render_cursor()

    def _render_cavern(self):
        # Collector
        collector_pos = pygame.Vector2(self.GRID_SIZE_X / 2, self.GRID_SIZE_Y - 2)
        self._render_iso_rect(collector_pos, 2, 2, self.COLOR_COLLECTOR, is_floor=True)

        # Spawner area
        spawner_pos = pygame.Vector2(self.GRID_SIZE_X / 2 - 1, 1)
        self._render_iso_rect(spawner_pos, 2, 2, self.COLOR_SPAWNER, is_floor=True)
        
        # Walls
        for i in range(self.GRID_SIZE_X):
            self._render_iso_rect(pygame.Vector2(i, -1), 1, 1, self.COLOR_WALL)
            self._render_iso_rect(pygame.Vector2(i, self.GRID_SIZE_Y - 1), 1, 1, self.COLOR_WALL)
        for i in range(self.GRID_SIZE_Y):
            self._render_iso_rect(pygame.Vector2(-1, i), 1, 1, self.COLOR_WALL)
            self._render_iso_rect(pygame.Vector2(self.GRID_SIZE_X - 1, i), 1, 1, self.COLOR_WALL)

        # Grid lines
        for y in range(self.GRID_SIZE_Y):
            start_pos = self._iso_to_screen(0, y)
            end_pos = self._iso_to_screen(self.GRID_SIZE_X, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos)
        for x in range(self.GRID_SIZE_X + 1):
            start_pos = self._iso_to_screen(x, 0)
            end_pos = self._iso_to_screen(x, self.GRID_SIZE_Y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos)

    def _render_iso_rect(self, pos, w, h, color, is_floor=False):
        height = 0 if is_floor else 50
        p1 = self._iso_to_screen(pos.x, pos.y, 0)
        p2 = self._iso_to_screen(pos.x + w, pos.y, 0)
        p3 = self._iso_to_screen(pos.x + w, pos.y + h, 0)
        p4 = self._iso_to_screen(pos.x, pos.y + h, 0)
        
        if is_floor:
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], color)
        else:
            # Draw sides to give illusion of height
            top_color = color
            side_color_dark = tuple(max(0, c - 30) for c in color)
            side_color_light = tuple(max(0, c - 15) for c in color)
            
            # Top
            pygame.gfxdraw.filled_polygon(self.screen, [
                self._iso_to_screen(pos.x, pos.y, height),
                self._iso_to_screen(pos.x+w, pos.y, height),
                self._iso_to_screen(pos.x+w, pos.y+h, height),
                self._iso_to_screen(pos.x, pos.y+h, height)
            ], top_color)
            # Left Side
            pygame.gfxdraw.filled_polygon(self.screen, [p3, p4, self._iso_to_screen(pos.x, pos.y+h, height), self._iso_to_screen(pos.x+w, pos.y+h, height)], side_color_light)
            # Right Side
            pygame.gfxdraw.filled_polygon(self.screen, [p2, p3, self._iso_to_screen(pos.x+w, pos.y+h, height), self._iso_to_screen(pos.x+w, pos.y, height)], side_color_dark)


    def _render_crystal(self, crystal):
        color_name = crystal['color']
        base_color = self.ORB_COLORS[color_name]
        
        # Semi-transparent fill
        s = pygame.Surface((self.TILE_WIDTH_HALF*2, self.TILE_HEIGHT_HALF*2), pygame.SRCALPHA)
        points = [
            (self.TILE_WIDTH_HALF, 0),
            (self.TILE_WIDTH_HALF * 2, self.TILE_HEIGHT_HALF),
            (self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF * 2),
            (0, self.TILE_HEIGHT_HALF)
        ]
        pygame.gfxdraw.filled_polygon(s, points, (*base_color, 80))
        
        sx, sy = self._iso_to_screen(crystal['pos'].x, crystal['pos'].y)
        self.screen.blit(s, (sx - self.TILE_WIDTH_HALF, sy - self.TILE_HEIGHT_HALF))

        # Bright outline
        p1 = self._iso_to_screen(crystal['pos'].x, crystal['pos'].y)
        p2 = self._iso_to_screen(crystal['pos'].x + 1, crystal['pos'].y)
        p3 = self._iso_to_screen(crystal['pos'].x + 1, crystal['pos'].y + 1)
        p4 = self._iso_to_screen(crystal['pos'].x, crystal['pos'].y + 1)
        pygame.draw.lines(self.screen, base_color, True, [p1, p2, p3, p4], 2)

    def _render_orb(self, orb, sx, sy):
        color = self.ORB_COLORS[orb['color']]
        radius = int(self.ORB_RADIUS * (1 + orb['pos'].z / 400))
        
        # Shadow
        shadow_x, shadow_y = self._iso_to_screen(orb['pos'].x, orb['pos'].y)
        shadow_radius = int(self.ORB_RADIUS * 0.8)
        shadow_alpha = max(0, 100 - orb['pos'].z * 0.5)
        if shadow_alpha > 0:
            pygame.gfxdraw.filled_circle(self.screen, shadow_x, shadow_y, shadow_radius, (0, 0, 0, int(shadow_alpha)))

        # Glow effect
        glow_color = (*color, 60)
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius + 3, glow_color)
        
        # Orb body
        pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, color)

        # Highlight
        highlight_color = (255, 255, 255, 150)
        pygame.gfxdraw.filled_circle(self.screen, sx - radius//3, sy - radius//3, radius//3, highlight_color)

    def _render_particle(self, particle, sx, sy):
        alpha = max(0, (particle['lifespan'] / 30.0) * 255)
        color = (*particle['color'], int(alpha))
        size = int(max(1, (particle['lifespan'] / 30.0) * 4))
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, size, color)

    def _render_cursor(self):
        color_name = self.available_crystal_colors[self.selected_crystal_color_idx]
        color = self.ORB_COLORS[color_name]

        p1 = self._iso_to_screen(self.cursor_pos.x, self.cursor_pos.y)
        p2 = self._iso_to_screen(self.cursor_pos.x + 1, self.cursor_pos.y)
        p3 = self._iso_to_screen(self.cursor_pos.x + 1, self.cursor_pos.y + 1)
        p4 = self._iso_to_screen(self.cursor_pos.x, self.cursor_pos.y + 1)
        
        pygame.draw.lines(self.screen, color, True, [p1, p2, p3, p4], 2)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}/{self.WIN_SCORE}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # Escaped
        escaped_text = self.font_small.render(f"ESCAPED: {self.escaped_orbs}/{self.MAX_ESCAPED}", True, (255, 100, 100))
        self.screen.blit(escaped_text, (self.SCREEN_WIDTH - escaped_text.get_width() - 10, 10))

        # Selected Crystal
        color_name = self.available_crystal_colors[self.selected_crystal_color_idx]
        color = self.ORB_COLORS[color_name]
        pygame.draw.rect(self.screen, color, (10, 35, 20, 20))
        pygame.draw.rect(self.screen, (255,255,255), (10, 35, 20, 20), 1)
        selected_text = self.font_small.render("Crystal", True, (255, 255, 255))
        self.screen.blit(selected_text, (35, 36))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.win else "GAME OVER"
            end_color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_large.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "escaped_orbs": self.escaped_orbs,
            "win": self.win,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly, but requires a display.
    # It will fail in a headless environment.
    # To run, you might need to comment out the `os.environ` line at the top.
    
    # Re-enable display for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Orb Cavern")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while not done:
        # --- Human Input ---
        movement = 0 # none
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = np.array([movement, space, shift])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Win: {info['win']}")
    pygame.time.wait(2000)
    env.close()