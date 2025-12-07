import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:49:47.578064
# Source Brief: brief_00643.md
# Brief Index: 643
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Navigate a soundscape by masking your movement with matching audio frequencies,
    using sound effect generators from your inventory.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a sound-sensitive maze by matching ambient audio frequencies to mask your movement from detection sensors."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press Shift to cycle through sound types and Space to activate a sound to mask your movement."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    TILE_SIZE = 40
    GRID_W, GRID_H = WIDTH // TILE_SIZE, HEIGHT // TILE_SIZE
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (15, 18, 28)
    COLOR_WALL = (40, 45, 60)
    COLOR_EXIT = (255, 220, 0)
    COLOR_PLAYER = (200, 255, 255)
    COLOR_SENSOR = (255, 20, 60)
    
    SOUND_COLORS = {
        1: (0, 100, 255),  # Low Freq - Blue
        2: (80, 255, 120), # Mid Freq - Green
        3: (255, 0, 255),  # High Freq - Magenta
    }
    
    PLAYER_SOUND_COLOR = (0, 255, 100) # Player sound is always green for clarity

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Exact spaces as required
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_visual_pos = [0.0, 0.0]
        self.exit_pos = [0, 0]
        self.walls = []
        self.sensors = []
        self.ambient_frequency = 1
        self.ambient_change_interval = 50
        self.player_inventory = []
        self.selected_inventory_index = 0
        self.active_player_sound = None
        self.detection_level = 0.0
        self.last_dist_to_exit = 0
        self.particles = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.detection_level = 0.0
        self.active_player_sound = None
        self.selected_inventory_index = 0
        self.particles = []
        
        self._generate_level()

        self.player_visual_pos = [float(self.player_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2),
                                  float(self.player_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)]
        self.last_dist_to_exit = self._get_dist_to_exit()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.walls = set()
        for x in range(self.GRID_W):
            self.walls.add((x, 0))
            self.walls.add((x, self.GRID_H - 1))
        for y in range(self.GRID_H):
            self.walls.add((0, y))
            self.walls.add((self.GRID_W - 1, y))

        # Simple maze-like structure
        for i in range(int(self.GRID_W * self.GRID_H * 0.2)):
            x = self.np_random.integers(1, self.GRID_W - 2)
            y = self.np_random.integers(1, self.GRID_H - 2)
            self.walls.add((x, y))

        def get_random_empty_pos():
            while True:
                x = self.np_random.integers(1, self.GRID_W - 1)
                y = self.np_random.integers(1, self.GRID_H - 1)
                if (x, y) not in self.walls:
                    return [x, y]

        self.player_pos = get_random_empty_pos()
        self.exit_pos = get_random_empty_pos()
        
        # Ensure player and exit are not on the same spot
        while self._get_dist_to_exit() < 3:
            self.exit_pos = get_random_empty_pos()

        # Place sensors
        self.sensors = []
        for _ in range(3):
            pos = get_random_empty_pos()
            # Ensure sensors are not on player or exit
            while pos == self.player_pos or pos == self.exit_pos:
                pos = get_random_empty_pos()
            
            radius = self.np_random.integers(3, 6) * self.TILE_SIZE
            self.sensors.append({"pos": pos, "range": radius})

        self.player_inventory = [1, 2, 3] # Player has all sound types
        self.ambient_frequency = self.np_random.integers(1, 4) # Use integers for [1,2,3]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- ACTION HANDLING ---
        # 1. Cycle inventory (Shift)
        if shift_press:
            self.selected_inventory_index = (self.selected_inventory_index + 1) % len(self.player_inventory)
            # sfx: UI_Cycle

        # 2. Activate sound (Space)
        if space_press and self.active_player_sound is None:
            freq = self.player_inventory[self.selected_inventory_index]
            self.active_player_sound = {"freq": freq, "duration": 5, "visual_radius": 0}
            # sfx: Player_Sound_Activate

        # --- GAME LOGIC UPDATE ---
        self._update_particles()
        if self.active_player_sound:
            self.active_player_sound["duration"] -= 1
            if self.active_player_sound["duration"] <= 0:
                self.active_player_sound = None

        # Change ambient sound periodically
        if self.steps > 0 and self.steps % self.ambient_change_interval == 0:
            self.ambient_frequency = self.np_random.choice([f for f in [1, 2, 3] if f != self.ambient_frequency])
            # sfx: Ambient_Shift

        # 3. Handle Movement
        prev_pos = self.player_pos[:]
        target_pos = self.player_pos[:]
        if movement == 1: target_pos[1] -= 1 # Up
        elif movement == 2: target_pos[1] += 1 # Down
        elif movement == 3: target_pos[0] -= 1 # Left
        elif movement == 4: target_pos[0] += 1 # Right

        if tuple(target_pos) not in self.walls:
            self.player_pos = target_pos
        
        moved = prev_pos != self.player_pos

        # --- DETECTION & REWARD CALCULATION ---
        is_masked = self.active_player_sound is not None and self.active_player_sound["freq"] == self.ambient_frequency
        detection_this_step = 0
        
        if moved:
            for sensor in self.sensors:
                sensor_px_pos = [sensor["pos"][0] * self.TILE_SIZE + self.TILE_SIZE/2, sensor["pos"][1] * self.TILE_SIZE + self.TILE_SIZE/2]
                player_px_pos = [self.player_pos[0] * self.TILE_SIZE + self.TILE_SIZE/2, self.player_pos[1] * self.TILE_SIZE + self.TILE_SIZE/2]
                dist = math.hypot(player_px_pos[0] - sensor_px_pos[0], player_px_pos[1] - sensor_px_pos[1])

                if dist < sensor["range"]:
                    if is_masked:
                        reward += 5.0 # Masked movement near sensor
                        self._create_particles(20, self.player_visual_pos, self.PLAYER_SOUND_COLOR)
                        # sfx: Mask_Success
                    else:
                        detection_increase = (1.0 - (dist / sensor["range"])) * 0.2
                        detection_this_step += detection_increase
                        reward -= 10.0 # Unmasked movement near sensor
                        self._create_particles(20, self.player_visual_pos, self.COLOR_SENSOR)
                        # sfx: Detection_Pulse
        
        self.detection_level = min(1.0, self.detection_level + detection_this_step)
        if detection_this_step > 0:
            reward -= 0.1 # Small penalty for any detection increase

        # Distance-based reward
        new_dist_to_exit = self._get_dist_to_exit()
        if new_dist_to_exit < self.last_dist_to_exit:
            reward += 1.0
        self.last_dist_to_exit = new_dist_to_exit
        
        self.steps += 1
        self.score += reward
        
        # --- TERMINATION CHECK ---
        terminated = False
        truncated = False
        if self.player_pos == self.exit_pos:
            reward = 100.0 # Override reward for win
            self.score += reward
            terminated = True
            self.game_over = True
            # sfx: Level_Win
        elif self.detection_level >= 1.0:
            reward = -50.0 # Override reward for loss
            self.score = -50.0 # Set score for loss
            terminated = True
            self.game_over = True
            # sfx: Alarm_Fail
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Interpolate player visual position for smooth movement
        target_px = [self.player_pos[0] * self.TILE_SIZE + self.TILE_SIZE/2, self.player_pos[1] * self.TILE_SIZE + self.TILE_SIZE/2]
        self.player_visual_pos[0] += (target_px[0] - self.player_visual_pos[0]) * 0.25
        self.player_visual_pos[1] += (target_px[1] - self.player_visual_pos[1]) * 0.25

        self._render_ambient_sound()
        self._render_walls()
        self._render_exit()
        self._render_sensors()
        self._render_player_sound()
        self._render_particles()
        self._render_player()

    def _render_walls(self):
        for x, y in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

    def _render_exit(self):
        pos_px = (self.exit_pos[0] * self.TILE_SIZE, self.exit_pos[1] * self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (pos_px[0], pos_px[1], self.TILE_SIZE, self.TILE_SIZE))
        # Add a glow effect
        glow_surf = pygame.Surface((self.TILE_SIZE*2, self.TILE_SIZE*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_EXIT, 30), (self.TILE_SIZE, self.TILE_SIZE), self.TILE_SIZE)
        self.screen.blit(glow_surf, (pos_px[0] - self.TILE_SIZE/2, pos_px[1] - self.TILE_SIZE/2), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ambient_sound(self):
        color = self.SOUND_COLORS[self.ambient_frequency]
        num_lines = 5
        for i in range(num_lines):
            y_offset = (i + 1) * (self.HEIGHT / (num_lines + 1))
            alpha = 10 + (math.sin(self.steps * 0.05 + i) * 0.5 + 0.5) * 20
            pygame.draw.line(self.screen, (*color, int(alpha)), (0, y_offset), (self.WIDTH, y_offset), 2)

    def _render_sensors(self):
        for sensor in self.sensors:
            pos_px = (int(sensor["pos"][0] * self.TILE_SIZE + self.TILE_SIZE / 2),
                      int(sensor["pos"][1] * self.TILE_SIZE + self.TILE_SIZE / 2))
            
            # Pulsating range visualization
            pulse = math.sin(self.steps * 0.1) * 0.5 + 0.5
            alpha = int(5 + pulse * 25)
            radius = int(sensor["range"])
            
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, (*self.COLOR_SENSOR, alpha))
            
            # Core sensor
            pygame.draw.circle(self.screen, self.COLOR_SENSOR, pos_px, 5)

    def _render_player(self):
        pos_px = (int(self.player_visual_pos[0]), int(self.player_visual_pos[1]))
        
        # Glow effect
        for i in range(3):
            alpha = 60 - i * 20
            radius = 10 + i * 5
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], radius, (*self.COLOR_PLAYER, alpha))
            pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, (*self.COLOR_PLAYER, alpha))

        # Player core
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, pos_px, 8)

    def _render_player_sound(self):
        if self.active_player_sound:
            pos_px = (int(self.player_visual_pos[0]), int(self.player_visual_pos[1]))
            # Animate the sound wave expanding
            self.active_player_sound["visual_radius"] += 20
            radius = int(self.active_player_sound["visual_radius"])
            
            # Fade out as it expands and as duration runs out
            max_radius = 200
            alpha_expand = max(0, 1 - (radius / max_radius))
            alpha_duration = self.active_player_sound["duration"] / 5.0
            alpha = int(alpha_expand * alpha_duration * 100)
            
            if radius > 0 and alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, (*self.PLAYER_SOUND_COLOR, alpha))
                pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius-1, (*self.PLAYER_SOUND_COLOR, alpha))

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, (*p['color'], p['alpha']), (int(p['x']), int(p['y'])), int(p['size']))

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['alpha'] -= 5
            p['size'] = max(0, p['size'] - 0.1)
            if p['alpha'] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _create_particles(self, count, pos, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'x': pos[0], 'y': pos[1],
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'size': self.np_random.uniform(2, 5),
                'alpha': self.np_random.integers(100, 255),
                'color': color
            })

    def _render_ui(self):
        # --- Detection Bar ---
        bar_w, bar_h = 200, 20
        bar_x, bar_y = (self.WIDTH - bar_w) / 2, self.HEIGHT - bar_h - 10
        fill_w = self.detection_level * bar_w
        
        # Color interpolation from green to red
        bar_color = (int(self.detection_level * 255), int((1 - self.detection_level) * 200), 0)
        
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(self.screen, (150, 150, 150), (bar_x, bar_y, bar_w, bar_h), 1)
        
        # --- Inventory Display ---
        inv_w, inv_h = 40, 40
        total_inv_w = len(self.player_inventory) * (inv_w + 5) - 5
        start_x = (self.WIDTH - total_inv_w) / 2
        
        for i, freq in enumerate(self.player_inventory):
            x = start_x + i * (inv_w + 5)
            y = 10
            color = self.SOUND_COLORS[freq]
            pygame.draw.rect(self.screen, color, (x, y, inv_w, inv_h), border_radius=5)
            if i == self.selected_inventory_index:
                pygame.draw.rect(self.screen, (255, 255, 255), (x-2, y-2, inv_w+4, inv_h+4), 2, border_radius=7)
        
        # --- Text Info ---
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, (200, 200, 200))
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, (200, 200, 200))
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "detection_level": self.detection_level,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos
        }

    def _get_dist_to_exit(self):
        return abs(self.player_pos[0] - self.exit_pos[0]) + abs(self.player_pos[1] - self.exit_pos[1])

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It will not work in a headless environment
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass # It was not set, which is fine
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Soundscape Stealth")
    
    done = False
    clock = pygame.time.Clock()
    
    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert the observation (H, W, C) to a Pygame surface (W, H)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause for 2 seconds before restarting
            
        clock.tick(30) # Limit to 30 FPS
        
    env.close()