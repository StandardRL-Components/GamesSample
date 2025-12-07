import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:47:46.044642
# Source Brief: brief_01289.md
# Brief Index: 1289
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Repair ancient underwater structures by matching falling colored tiles to their sockets. "
        "Use your time-slowing ability to place them perfectly before the timer runs out."
    )
    user_guide = (
        "Controls: ←→ to move, ↑↓ to cycle color, space to hard-drop, and shift to slow time."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2500

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Colors & Visuals ---
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_RUINS = (60, 70, 90)
        self.COLOR_RUINS_OUTLINE = (80, 90, 110)
        self.TILE_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
        ]
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_TIME_SLOW_OVERLAY = (200, 220, 255, 40)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.timer = 0
        self.time_slows_available = 0
        self.time_slow_duration = 0
        self.tile_fall_speed_base = 0
        self.current_tile = None
        self.placed_tiles = []
        self.structures = []
        self.particles = []
        self.plankton = []
        self.previous_space_held = False
        self.previous_shift_held = False
        self.rotation_cooldown = 0
        self.reward_this_step = 0
        
        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        
        self.timer = int(90 * self.FPS) # 90 seconds
        self.time_slows_available = 3
        self.time_slow_duration = 0
        
        self.tile_fall_speed_base = 1.0
        self.placed_tiles = []
        self.particles = []
        self.plankton = self._generate_plankton(100)
        
        self.structures = self._generate_level(1)
        self._spawn_new_tile()

        self.previous_space_held = False
        self.previous_shift_held = False
        self.rotation_cooldown = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = -0.001 # Small penalty for time passing

        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        self._handle_input(movement, space_held, shift_held)
        self._update_game_state()
        
        self.steps += 1
        
        terminated = self._check_termination()
        reward = self.reward_this_step

        if terminated:
            if self.win_condition:
                reward += 50.0 # Goal-oriented reward for winning
            else:
                reward -= 50.0 # Goal-oriented penalty for losing

        # The 'truncated' flag is False as termination is based on game state, not an artificial step limit.
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        if self.current_tile is None:
            return

        # --- Horizontal Movement ---
        move_speed = 7
        if movement == 3: # Left
            self.current_tile['x'] = max(self.current_tile['size'] // 2, self.current_tile['x'] - move_speed)
        elif movement == 4: # Right
            self.current_tile['x'] = min(self.WIDTH - self.current_tile['size'] // 2, self.current_tile['x'] + move_speed)

        # --- Tile Type "Rotation" ---
        if self.rotation_cooldown > 0:
            self.rotation_cooldown -= 1
        
        if self.rotation_cooldown == 0:
            if movement == 1: # Up -> Cycle type backwards
                self.current_tile['type'] = (self.current_tile['type'] - 1) % len(self.TILE_COLORS)
                self.rotation_cooldown = 5 # 5-frame cooldown
            elif movement == 2: # Down -> Cycle type forwards
                self.current_tile['type'] = (self.current_tile['type'] + 1) % len(self.TILE_COLORS)
                self.rotation_cooldown = 5

        # --- Hard Drop (on press) ---
        if space_held and not self.previous_space_held:
            landing_y = self._get_landing_y()
            self.current_tile['y'] = landing_y
            self._lock_tile_and_check_matches()

        # --- Time Slow (on press) ---
        if shift_held and not self.previous_shift_held:
            if self.time_slows_available > 0 and self.time_slow_duration <= 0:
                self.time_slows_available -= 1
                self.time_slow_duration = int(4 * self.FPS) # 4 seconds
                self.reward_this_step += 0.5 # Small reward for using the ability
                for _ in range(50):
                    self._create_particle(self.WIDTH/2, self.HEIGHT/2, (255,255,255), 10, 2, 360)


        self.previous_space_held = space_held
        self.previous_shift_held = shift_held

    def _update_game_state(self):
        # --- Update Timer ---
        if self.time_slow_duration > 0:
            self.time_slow_duration -= 1
        else:
            self.timer = max(0, self.timer - 1)
        
        # --- Update Tile Fall Speed ---
        speed_multiplier = 0.2 if self.time_slow_duration > 0 else 1.0
        difficulty_increase = (self.steps // 500) * 0.1
        current_fall_speed = (self.tile_fall_speed_base + difficulty_increase) * speed_multiplier

        # --- Update Falling Tile Position ---
        if self.current_tile:
            self.current_tile['y'] += current_fall_speed
            if self.current_tile['y'] >= self._get_landing_y():
                self.current_tile['y'] = self._get_landing_y()
                self._lock_tile_and_check_matches()

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['size'] = max(0, p['size'] * 0.95)
        
        # --- Update Plankton ---
        for p in self.plankton:
            p['y'] -= p['speed']
            if p['y'] < -5:
                p['y'] = self.HEIGHT + 5
                p['x'] = self.np_random.uniform(0, self.WIDTH)


    def _get_landing_y(self):
        if self.current_tile is None: return self.HEIGHT - 10
        
        tile_x = self.current_tile['x']
        tile_size = self.current_tile['size']
        
        # Check against floor
        max_y = self.HEIGHT - tile_size // 2

        # Check against placed tiles
        for pt in self.placed_tiles:
            if abs(tile_x - pt['x']) < (tile_size + pt['size']) // 2:
                if pt['y'] - tile_size // 2 < max_y:
                    max_y = pt['y'] - tile_size // 2 -1

        return max_y

    def _lock_tile_and_check_matches(self):
        if self.current_tile is None: return

        locked_tile = self.current_tile
        self.placed_tiles.append(locked_tile)
        self.current_tile = None
        
        # Create splash particles
        for _ in range(15):
            self._create_particle(locked_tile['x'], locked_tile['y'], self.TILE_COLORS[locked_tile['type']], 3, 1, 180, 180)

        # Check for matches
        match_found = False
        for struct in self.structures:
            is_struct_complete_before = all(s['repaired'] for s in struct)
            for socket in struct:
                if not socket['repaired']:
                    dist = math.hypot(locked_tile['x'] - socket['x'], locked_tile['y'] - socket['y'])
                    if dist < 35 and locked_tile['type'] == socket['type']:
                        socket['repaired'] = True
                        match_found = True
                        self.reward_this_step += 1.0 # Reward for a correct match
                        self.score += 10
                        for _ in range(30):
                             self._create_particle(socket['x'], socket['y'], self.TILE_COLORS[socket['type']], 5, 2)
            
            is_struct_complete_after = all(s['repaired'] for s in struct)
            if is_struct_complete_after and not is_struct_complete_before:
                self.reward_this_step += 5.0 # Bonus for completing a structure
                self.score += 100

        if not match_found:
             self.reward_this_step -= 0.5 # Penalty for a useless placement

        # Check for win condition
        if all(s['repaired'] for struct in self.structures for s in struct):
            self.win_condition = True
            self.game_over = True
        else:
            self._spawn_new_tile()

    def _spawn_new_tile(self):
        tile_size = 24
        # Ensure new tiles don't spawn on top of existing ones
        spawn_y = tile_size
        for pt in self.placed_tiles:
            if pt['y'] < spawn_y * 2: # Check if any tile is in the spawn area
                self.game_over = True # Topped out
                return
        
        required_types = {s['type'] for struct in self.structures for s in struct if not s['repaired']}
        if not required_types: # All repaired, but win condition not triggered yet
            spawn_type = self.np_random.integers(0, len(self.TILE_COLORS))
        else:
            spawn_type = self.np_random.choice(list(required_types))


        self.current_tile = {
            'x': self.WIDTH // 2,
            'y': spawn_y,
            'type': spawn_type,
            'size': tile_size
        }
        self.rotation_cooldown = 10 # Grace period before rotation is allowed

    def _check_termination(self):
        if self.game_over:
            return True
        if self.timer <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_structures()
        self._render_placed_tiles()
        if self.current_tile:
            self._render_tile(self.current_tile, is_current=True)
        self._render_particles()
        if self.time_slow_duration > 0:
            self._render_time_slow_overlay()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "time_slows": self.time_slows_available,
            "level_complete": self.win_condition,
        }

    # --- Rendering Methods ---
    
    def _render_tile(self, tile, is_current=False):
        x, y, color_idx, size = int(tile['x']), int(tile['y']), tile['type'], tile['size']
        color = self.TILE_COLORS[color_idx]
        
        rect = pygame.Rect(x - size//2, y - size//2, size, size)
        
        # Draw main tile
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Glow effect for current tile
        if is_current:
            glow_color = color
            glow_surface = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*glow_color, 60), (size, size), size)
            pygame.draw.circle(glow_surface, (*glow_color, 30), (size, size), int(size*0.75))
            self.screen.blit(glow_surface, (x - size, y - size), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Draw landing projection
            landing_y = int(self._get_landing_y())
            pygame.draw.rect(self.screen, (*color, 60), (x - size//2, landing_y - size//2, size, size), 2, border_radius=4)


    def _render_structures(self):
        for struct in self.structures:
            for socket in struct:
                x, y, color_idx = int(socket['x']), int(socket['y']), socket['type']
                color = self.TILE_COLORS[color_idx]
                size = 14
                
                if socket['repaired']:
                    # Glowing repaired socket
                    glow_color = color
                    pygame.gfxdraw.filled_circle(self.screen, x, y, size, glow_color)
                    pygame.gfxdraw.aacircle(self.screen, x, y, size, glow_color)
                    
                    # Additive blend glow
                    glow_surface = pygame.Surface((size*4, size*4), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surface, (*glow_color, 80), (size*2, size*2), size*2)
                    self.screen.blit(glow_surface, (x - size*2, y - size*2), special_flags=pygame.BLEND_RGBA_ADD)
                else:
                    # Empty socket
                    pygame.gfxdraw.aacircle(self.screen, x, y, size, self.COLOR_RUINS_OUTLINE)
                    pygame.gfxdraw.aacircle(self.screen, x, y, size-1, self.COLOR_RUINS_OUTLINE)
                    pygame.gfxdraw.aacircle(self.screen, x, y, size-2, color)


    def _render_placed_tiles(self):
        for tile in self.placed_tiles:
            self._render_tile(tile)

    def _render_background_effects(self):
        for p in self.plankton:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p['x']-p['size'], p['y']-p['size']), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_time_slow_overlay(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill(self.COLOR_TIME_SLOW_OVERLAY)
        self.screen.blit(overlay, (0,0))
    
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:06d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left_sec = self.timer / self.FPS
        timer_color = (255, 100, 100) if time_left_sec < 15 else self.COLOR_UI_TEXT
        timer_text = self.font_ui.render(f"TIME: {time_left_sec:04.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        # Time Slow Charges
        charge_text = self.font_ui.render("TIME SLOW:", True, self.COLOR_UI_TEXT)
        self.screen.blit(charge_text, (10, 35))
        for i in range(self.time_slows_available):
            pygame.draw.circle(self.screen, (200, 220, 255), (130 + i * 25, 45), 8)
            pygame.draw.circle(self.screen, self.COLOR_BG, (130 + i * 25, 45), 5)


    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        message = "LEVEL COMPLETE" if self.win_condition else "TIME UP"
        color = (100, 255, 100) if self.win_condition else (255, 100, 100)
        
        text = self.font_game_over.render(message, True, color)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text, text_rect)

    # --- Helper Methods ---

    def _generate_level(self, level_num):
        # A simple level generator
        struct1_sockets = [
            {'x': 120, 'y': 360, 'type': 0, 'repaired': False},
            {'x': 160, 'y': 360, 'type': 1, 'repaired': False},
            {'x': 200, 'y': 360, 'type': 0, 'repaired': False},
        ]
        struct2_sockets = [
            {'x': 440, 'y': 360, 'type': 2, 'repaired': False},
            {'x': 480, 'y': 360, 'type': 3, 'repaired': False},
            {'x': 520, 'y': 360, 'type': 2, 'repaired': False},
        ]
        return [struct1_sockets, struct2_sockets]
    
    def _generate_plankton(self, count):
        plankton = []
        for _ in range(count):
            plankton.append({
                'x': self.np_random.uniform(0, self.WIDTH),
                'y': self.np_random.uniform(0, self.HEIGHT),
                'size': self.np_random.uniform(0.5, 2),
                'speed': self.np_random.uniform(0.1, 0.5),
                'color': (self.np_random.integers(30, 60), self.np_random.integers(50, 80), self.np_random.integers(70, 100))
            })
        return plankton

    def _create_particle(self, x, y, color, size, speed, angle_spread=360, angle_offset=0):
        angle = math.radians(self.np_random.uniform(0, angle_spread) + angle_offset)
        vel = self.np_random.uniform(0.5, 1.0) * speed
        self.particles.append({
            'x': x, 'y': y,
            'vx': math.cos(angle) * vel, 'vy': math.sin(angle) * vel,
            'size': self.np_random.uniform(0.8, 1.2) * size,
            'life': self.np_random.integers(20, 40),
            'max_life': 40,
            'color': color
        })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # Ensure you have 'pygame' installed: pip install pygame
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    running = True
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Sunken City Tile Matcher")
    
    total_reward = 0
    
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1 # Rotate Left
        if keys[pygame.K_DOWN]: movement = 2 # Rotate Right
        if keys[pygame.K_LEFT]: movement = 3 # Move Left
        if keys[pygame.K_RIGHT]: movement = 4 # Move Right
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # The env will auto-handle game-over state, but we can wait for a reset key
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    env.close()