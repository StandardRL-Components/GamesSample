import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a shifting dreamscape, collect resources to solve puzzles, and slow time to avoid hazards."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to slow time and shift to interact with objects (collect resources or solve puzzles)."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 2000
    PUZZLES_TO_WIN = 10

    # Colors
    COLOR_BG = (20, 10, 40)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255)
    COLOR_RESOURCE = (0, 100, 255)
    COLOR_RESOURCE_GLOW = (50, 150, 255)
    COLOR_PUZZLE = (0, 255, 100)
    COLOR_PUZZLE_GLOW = (50, 255, 150)
    COLOR_PUZZLE_SOLVED = (100, 100, 100)
    COLOR_HAZARD = (255, 50, 50)
    COLOR_HAZARD_GLOW = (255, 100, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TIME_SLOW_EFFECT = (200, 200, 255)

    # Game Parameters
    PLAYER_RADIUS = 12
    PLAYER_SPEED = 6.0
    PLAYER_INTERP_FACTOR = 0.2
    INTERACT_RADIUS = 40
    TIME_SLOW_DURATION = 150 # 5 seconds at 30fps
    TIME_SLOW_FACTOR = 0.3
    ROOM_SHIFT_INTERVAL = 750 # 25 seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.game_over = False
        self.win_status = "" # "WIN", "LOSS_RESOURCES", "LOSS_TIME"
        
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        
        self.resource_count = 0
        self.puzzles_solved = 0
        
        self.time_slow_timer = 0
        self.room_shift_timer = 0
        
        self.resources = []
        self.puzzles = []
        self.hazards = []
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Initialize state for the first time
        # self.reset() is called by the environment runner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.game_over = False
        self.win_status = ""
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        
        self.resource_count = 3
        self.puzzles_solved = 0
        
        self.time_slow_timer = 0
        self.room_shift_timer = self.ROOM_SHIFT_INTERVAL
        
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._setup_room()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Determine effective speed ---
        speed_multiplier = self.TIME_SLOW_FACTOR if self.time_slow_timer > 0 else 1.0
        current_speed = self.PLAYER_SPEED * speed_multiplier

        # --- Handle Actions ---
        # Movement
        target_vel = pygame.Vector2(0, 0)
        if movement == 1: target_vel.y = -current_speed
        elif movement == 2: target_vel.y = current_speed
        elif movement == 3: target_vel.x = -current_speed
        elif movement == 4: target_vel.x = current_speed
        self.player_vel.x += (target_vel.x - self.player_vel.x) * self.PLAYER_INTERP_FACTOR
        self.player_vel.y += (target_vel.y - self.player_vel.y) * self.PLAYER_INTERP_FACTOR

        # Time-slow (on press)
        if space_held and not self.prev_space_held and self.time_slow_timer <= 0:
            self.time_slow_timer = self.TIME_SLOW_DURATION
            self._create_particles(self.player_pos, self.COLOR_TIME_SLOW_EFFECT, 30, is_ring=True)

        # Interact (on press)
        if shift_held and not self.prev_shift_held:
            reward += self._handle_interaction()

        # --- Update Game State ---
        self.steps += 1
        self.room_shift_timer -= 1
        self.time_slow_timer = max(0, self.time_slow_timer - 1)

        # Player position
        self.player_pos += self.player_vel
        self._enforce_boundaries()

        # Hazard collision
        reward += self._check_hazard_collisions()

        # Room shift
        if self.room_shift_timer <= 0:
            self._setup_room()
            self.room_shift_timer = self.ROOM_SHIFT_INTERVAL

        # Update particles
        self._update_particles(speed_multiplier)

        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.puzzles_solved >= self.PUZZLES_TO_WIN:
            reward += 100.0
            terminated = True
            self.win_status = "WIN"
        elif self.resource_count <= 0 and not self._can_solve_any_puzzle():
            reward -= 100.0
            terminated = True
            self.win_status = "LOSS_RESOURCES"
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            if not terminated:
                self.win_status = "LOSS_TIME"
        
        self.game_over = terminated or truncated
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "puzzles_solved": self.puzzles_solved,
            "resources": self.resource_count,
            "steps": self.steps,
            "time_slow_active": self.time_slow_timer > 0,
        }

    # --- Helper Methods: Game Logic ---

    def _setup_room(self):
        self.resources.clear()
        self.puzzles.clear()
        self.hazards.clear()
        
        # Determine puzzle difficulty
        puzzles_needed = 1 + (self.puzzles_solved // 3)
        
        # Place entities, ensuring no overlaps
        placed_positions = [self.player_pos]
        
        for _ in range(self.np_random.integers(2, 5)):
            self.resources.append(self._get_valid_pos(placed_positions, 20))
        for _ in range(puzzles_needed):
            self.puzzles.append({'pos': self._get_valid_pos(placed_positions, 25), 'solved': False})
        for _ in range(self.np_random.integers(1, 4)):
            self.hazards.append({'pos': self._get_valid_pos(placed_positions, 25), 'phase': self.np_random.uniform(0, math.pi * 2)})

    def _get_valid_pos(self, existing_positions, min_dist):
        while True:
            pos = pygame.Vector2(
                self.np_random.integers(50, self.SCREEN_WIDTH - 50),
                self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
            )
            if all(pos.distance_to(p) > min_dist for p in existing_positions):
                existing_positions.append(pos)
                return pos

    def _handle_interaction(self):
        # Find nearest resource
        nearest_res, dist_res = None, float('inf')
        for res_pos in self.resources:
            d = self.player_pos.distance_to(res_pos)
            if d < dist_res:
                dist_res = d
                nearest_res = res_pos
        
        # Find nearest unsolved puzzle
        nearest_puz, dist_puz = None, float('inf')
        for p in self.puzzles:
            if not p['solved']:
                d = self.player_pos.distance_to(p['pos'])
                if d < dist_puz:
                    dist_puz = d
                    nearest_puz = p
        
        # Prioritize resource collection if closer
        if nearest_res and dist_res < dist_puz and dist_res < self.INTERACT_RADIUS:
            self.resources.remove(nearest_res)
            self.resource_count += 1
            self._create_particles(nearest_res, self.COLOR_RESOURCE, 20)
            return 0.1
        
        # Otherwise, try to solve puzzle
        elif nearest_puz and dist_puz < self.INTERACT_RADIUS:
            if self.resource_count > 0:
                self.resource_count -= 1
                self.puzzles_solved += 1
                nearest_puz['solved'] = True
                self._create_particles(nearest_puz['pos'], self.COLOR_PUZZLE, 40)
                return 1.0
            else:
                pass # Not enough resources
        return 0.0

    def _check_hazard_collisions(self):
        reward = 0.0
        for hazard in self.hazards:
            if self.player_pos.distance_to(hazard['pos']) < self.PLAYER_RADIUS + 10:
                if self.resource_count > 0:
                    self.resource_count -= 1
                    reward -= 0.5
                    self._create_particles(self.player_pos, self.COLOR_HAZARD, 15)
                    # Knockback
                    knockback_vec = (self.player_pos - hazard['pos']).normalize() * 5
                    self.player_vel += knockback_vec
                # Move hazard to avoid continuous drain
                hazard['pos'] = self._get_valid_pos([p['pos'] for p in self.puzzles] + self.resources, 50)
        return reward

    def _enforce_boundaries(self):
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

    def _can_solve_any_puzzle(self):
        return any(not p['solved'] for p in self.puzzles)

    def _create_particles(self, pos, color, count, is_ring=False):
        for _ in range(count):
            if is_ring:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 4)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            else:
                vel = pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3))
            
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(15, 31),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _update_particles(self, speed_multiplier):
        for p in self.particles[:]:
            p['pos'] += p['vel'] * speed_multiplier
            p['vel'] *= 0.95 # Drag
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    # --- Helper Methods: Rendering ---

    def _render_background(self):
        # Subtle shifting geometric patterns
        time_factor = self.steps * 0.005
        for i in range(10):
            phase = i * math.pi / 5
            x1 = self.SCREEN_WIDTH / 2 + math.sin(time_factor + phase) * 200
            y1 = self.SCREEN_HEIGHT / 2 + math.cos(time_factor * 0.7 + phase) * 150
            x2 = self.SCREEN_WIDTH / 2 + math.cos(-time_factor * 0.8 + phase) * 300
            y2 = self.SCREEN_HEIGHT / 2 + math.sin(-time_factor + phase) * 250
            pygame.gfxdraw.line(self.screen, int(x1), int(y1), int(x2), int(y2), (30, 20, 50))

    def _render_game_elements(self):
        # Render hazards
        for h in self.hazards:
            size = 10 + 3 * math.sin(self.steps * 0.1 + h['phase'])
            self._draw_glow_circle(h['pos'], size, self.COLOR_HAZARD, self.COLOR_HAZARD_GLOW)

        # Render resources
        for r_pos in self.resources:
            size = 8 + 2 * math.sin(self.steps * 0.05)
            self._draw_glow_circle(r_pos, size, self.COLOR_RESOURCE, self.COLOR_RESOURCE_GLOW)

        # Render puzzles
        for p in self.puzzles:
            if p['solved']:
                self._draw_glow_circle(p['pos'], 12, self.COLOR_PUZZLE_SOLVED, (80,80,80))
            else:
                size = 12 + 2 * math.cos(self.steps * 0.05)
                self._draw_glow_circle(p['pos'], size, self.COLOR_PUZZLE, self.COLOR_PUZZLE_GLOW)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 30))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

        # Render player
        self._draw_glow_circle(self.player_pos, self.PLAYER_RADIUS, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

        # Render time-slow effect
        if self.time_slow_timer > 0:
            progress = (self.TIME_SLOW_DURATION - self.time_slow_timer) / self.TIME_SLOW_DURATION
            radius = int(progress * self.PLAYER_RADIUS * 3)
            alpha = int(100 * (1 - progress))
            if alpha > 0:
                self._draw_circle_alpha(self.player_pos, radius, (*self.COLOR_TIME_SLOW_EFFECT, alpha))
                self._draw_circle_alpha(self.player_pos, radius+2, (*self.COLOR_TIME_SLOW_EFFECT, alpha//2))


    def _render_ui(self):
        # Resource Count
        res_text = self.font_small.render(f"RESOURCES: {self.resource_count}", True, self.COLOR_UI_TEXT)
        self.screen.blit(res_text, (10, 10))
        
        # Puzzles Solved
        puzzle_text = self.font_small.render(f"PUZZLES: {self.puzzles_solved}/{self.PUZZLES_TO_WIN}", True, self.COLOR_UI_TEXT)
        self.screen.blit(puzzle_text, (self.SCREEN_WIDTH - puzzle_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_status == "WIN":
                msg = "DREAM ESCAPED"
                color = self.COLOR_PUZZLE
            else:
                msg = "DREAM COLLAPSED"
                color = self.COLOR_HAZARD
                
            text_surface = self.font_large.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _draw_glow_circle(self, pos, radius, color, glow_color):
        # Draw a blurred glow effect
        for i in range(int(radius), 0, -2):
            alpha = int(100 * (1 - i / radius))
            pygame.gfxdraw.filled_circle(
                self.screen, int(pos.x), int(pos.y), int(radius + i), (*glow_color, alpha)
            )
        # Draw the main circle
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), color)

    def _draw_circle_alpha(self, pos, radius, color):
        if radius > 0:
            surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (radius, radius), radius, width=2)
            self.screen.blit(surf, (pos.x - radius, pos.y - radius))

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # --- Manual Play Example ---
    # This block will not run in the testing environment, but is useful for development.
    # It requires a display to be available.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    env = GameEnv(render_mode="rgb_array")
    
    # Use a separate display for rendering
    pygame.display.set_caption("Dreamscape Puzzle")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    
    running = True
    while running:
        # --- Action Mapping for Human Player ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Resetting Environment ---")
                obs, info = env.reset()
                total_reward = 0
                terminated = False
                truncated = False

        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Episode Finished. Total Reward: {total_reward:.2f}")
                print(f"Final Info: {info}")
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(GameEnv.FPS)

    env.close()