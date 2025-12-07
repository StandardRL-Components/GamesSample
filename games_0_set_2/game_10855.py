import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:11:01.362653
# Source Brief: brief_00855.md
# Brief Index: 855
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Chroma Shift: A physics-based puzzle game Gymnasium environment.

    The player controls a laser emitter to clear colored block-enemies from the screen.
    Lasers push enemies of a matching color. The goal is to clear all enemies
    by pushing them off-screen before running out of moves.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Color Selection (0=none, 1=Red, 2=Green, 3=Blue, 4=White)
    - actions[1]: Fire Laser (0=released, 1=pressed)
    - actions[2]: Unused (0=released, 1=held)

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - RGB image of the game.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Push colored blocks off-screen using a matching colored laser. "
        "Clear all blocks before running out of moves."
    )
    user_guide = (
        "Use keys 1-4 to select laser color (Red, Green, Blue, White). "
        "Press space to fire."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAY_AREA_PADDING = 50
    FPS = 30

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 45, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_WHITE = (255, 255, 255)
    
    COLORS = {
        "RED": (255, 80, 80),
        "GREEN": (80, 255, 80),
        "BLUE": (80, 80, 255)
    }
    COLOR_NAMES = list(COLORS.keys())
    LASER_COLORS = list(COLORS.values()) + [COLOR_WHITE]
    LASER_COLOR_NAMES = COLOR_NAMES + ["WHITE"]
    
    # Physics
    DRAG_FACTOR = 0.96
    LASER_FORCE = 15.0
    LASER_LIFETIME = 6 # frames
    
    # Game Rules
    MAX_EPISODE_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Game state variables
        self.level = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.moves_remaining = 0
        self.enemies = []
        self.particles = []
        self.lasers = []
        self.selected_color_index = 0
        self.prev_space_held = False
        self.emitter_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20)
        self.target_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.laser_direction = (self.target_pos - self.emitter_pos).normalize()

        self.reset()
        # self.validate_implementation() # This is a debug tool, not required by the API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.level += 1
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles.clear()
        self.lasers.clear()
        self.enemies.clear()

        # Difficulty scaling
        num_enemies = 3 + (self.level - 1) // 2
        self.moves_remaining = 5 + (self.level - 1) // 2
        
        # Spawn enemies
        spawn_area = pygame.Rect(
            self.PLAY_AREA_PADDING, self.PLAY_AREA_PADDING,
            self.SCREEN_WIDTH - 2 * self.PLAY_AREA_PADDING,
            self.SCREEN_HEIGHT / 2
        )
        for _ in range(num_enemies):
            while True:
                pos = pygame.Vector2(
                    self.np_random.uniform(spawn_area.left, spawn_area.right),
                    self.np_random.uniform(spawn_area.top, spawn_area.bottom)
                )
                size = self.np_random.uniform(15, 25)
                # Check for overlap
                if not any(e['pos'].distance_to(pos) < e['size'] + size for e in self.enemies):
                    break
            
            color_name = self.np_random.choice(self.COLOR_NAMES)
            self.enemies.append({
                "pos": pos,
                "vel": pygame.Vector2(0, 0),
                "size": size,
                "color_name": color_name,
                "color_rgb": self.COLORS[color_name]
            })

        self.prev_space_held = False
        self.selected_color_index = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        # --- 1. Handle Actions ---
        color_selection = action[0]
        space_held = action[1] == 1
        
        if 1 <= color_selection <= 4:
            self.selected_color_index = color_selection - 1

        fire_action = space_held and not self.prev_space_held
        if fire_action and self.moves_remaining > 0 and not self.game_over:
            self.moves_remaining -= 1
            # sfx: LaserFire.wav
            self.lasers.append({
                "start": self.emitter_pos,
                "end": self.target_pos,
                "dir": self.laser_direction,
                "color_name": self.LASER_COLOR_NAMES[self.selected_color_index],
                "color_rgb": self.LASER_COLORS[self.selected_color_index],
                "lifetime": self.LASER_LIFETIME,
                "hit_enemies": set()
            })
            self._create_particles(self.emitter_pos, self.LASER_COLORS[self.selected_color_index], 20, 3)

        self.prev_space_held = space_held

        # --- 2. Update Game Logic ---
        reward += self._update_physics()
        
        # --- 3. Check Termination Conditions ---
        if not self.game_over:
            if not self.enemies:
                # Win condition
                reward += 50
                self.game_over = True
                # sfx: LevelComplete.wav
            elif self.moves_remaining <= 0 and not self.lasers:
                # Loss condition
                reward -= 50
                self.game_over = True
                # sfx: LevelFail.wav
        
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated
            self._get_info()
        )

    def _update_physics(self):
        reward = 0
        
        # Update and check lasers
        laser_hit_something = False
        for laser in self.lasers[:]:
            laser['lifetime'] -= 1
            if laser['lifetime'] <= 0:
                self.lasers.remove(laser)
                continue

            for i, enemy in enumerate(self.enemies):
                if i in laser['hit_enemies']:
                    continue

                # Collision check: distance from point (enemy center) to line segment (laser)
                t = ((enemy['pos'] - laser['start']).dot(laser['dir'])) / laser['start'].distance_to(laser['end'])
                t = max(0, min(1, t))
                closest_point = laser['start'] + t * (laser['end'] - laser['start'])
                
                if closest_point.distance_to(enemy['pos']) < enemy['size']:
                    laser_hit_something = True
                    laser['hit_enemies'].add(i)
                    
                    is_match = (laser['color_name'] == 'WHITE' or laser['color_name'] == enemy['color_name'])
                    if is_match:
                        reward += 1.0 # Hit matching enemy
                        enemy['vel'] += laser['dir'] * self.LASER_FORCE
                        self._create_particles(enemy['pos'], enemy['color_rgb'], 15, 2)
                        # sfx: HitConfirm.wav
                    else:
                        reward -= 0.1 # Hit mismatch
                        # sfx: HitMismatch.wav
        
        if self.lasers and not laser_hit_something and self.lasers[0]['lifetime'] == self.LASER_LIFETIME - 1:
            reward -= 0.1 # Missed shot

        # Update enemies
        for enemy in self.enemies[:]:
            enemy['vel'] *= self.DRAG_FACTOR
            enemy['pos'] += enemy['vel']
            
            # Check if off-screen (destruction)
            if not self.screen.get_rect().collidepoint(enemy['pos']):
                reward += 5.0 # Destroyed enemy
                self.score += 10
                self._create_particles(enemy['pos'], enemy['color_rgb'], 50, 4)
                self.enemies.remove(enemy)
                # sfx: EnemyDestroyed.wav

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
        
        return reward

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
            "level": self.level,
            "moves_remaining": self.moves_remaining,
            "enemies_remaining": len(self.enemies)
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            color = p['color'] + (alpha,)
            size = int(p['size'] * (p['lifetime'] / p['max_lifetime']))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (int(p['pos'].x - size), int(p['pos'].y - size)), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw lasers
        for laser in self.lasers:
            self._draw_laser_beam(laser['start'], laser['end'], laser['color_rgb'], laser['lifetime'])

        # Draw emitter
        self._draw_emitter()
        
        # Draw enemies
        for enemy in self.enemies:
            pos_int = (int(enemy['pos'].x), int(enemy['pos'].y))
            size_int = int(enemy['size'])
            
            # Body
            body_rect = pygame.Rect(pos_int[0] - size_int, pos_int[1] - size_int, size_int * 2, size_int * 2)
            pygame.draw.rect(self.screen, enemy['color_rgb'], body_rect, border_radius=3)
            # Border
            border_color = tuple(max(0, c-50) for c in enemy['color_rgb'])
            pygame.draw.rect(self.screen, border_color, body_rect, width=3, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Level
        level_text = self.font_large.render(f"LEVEL: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH/2 - level_text.get_width()/2, 10))
        
        # Moves remaining
        moves_text = self.font_large.render(f"MOVES: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (10, 10))

        # Selected color indicator
        color_name = self.LASER_COLOR_NAMES[self.selected_color_index]
        color_rgb = self.LASER_COLORS[self.selected_color_index]
        color_text = self.font_small.render(f"LASER: {color_name}", True, color_rgb)
        self.screen.blit(color_text, (self.emitter_pos.x - color_text.get_width() / 2, self.emitter_pos.y + 15))

    def _draw_emitter(self):
        pos_int = (int(self.emitter_pos.x), int(self.emitter_pos.y))
        color = self.LASER_COLORS[self.selected_color_index]
        
        # Glow effect
        for i in range(4, 0, -1):
            alpha = 80 - i * 15
            radius = 8 + i * 4
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color + (alpha,))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, color + (alpha,))

        # Core
        pygame.draw.circle(self.screen, self.COLOR_WHITE, pos_int, 8)
        pygame.draw.circle(self.screen, color, pos_int, 6)

    def _draw_laser_beam(self, start, end, color, lifetime):
        alpha_scale = min(1.0, (self.LASER_LIFETIME - lifetime) / 2.0) # Fade in
        
        # Glow
        glow_alpha = int(100 * alpha_scale)
        pygame.draw.line(self.screen, color + (glow_alpha,), start, end, width=15)
        
        # Core beam
        core_alpha = int(255 * alpha_scale)
        pygame.draw.line(self.screen, self.COLOR_WHITE + (core_alpha,), start, end, width=7)
        pygame.draw.line(self.screen, color + (core_alpha,), start, end, width=4)

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "color": color,
                "lifetime": lifetime,
                "max_lifetime": lifetime,
                "size": self.np_random.uniform(2, 5)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage:
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part allows a human to play the game.
    # Controls: 1-4 to select color, SPACE to fire, Q to quit.
    
    # Re-initialize pygame for display
    pygame.display.set_caption("Chroma Shift")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Action buffer
    action = [0, 0, 0] # [color, fire, shift]
    
    running = True
    while running:
        # Reset action buffer at the start of the frame
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                
                # Color selection
                if event.key == pygame.K_1: action[0] = 1 # Red
                if event.key == pygame.K_2: action[0] = 2 # Green
                if event.key == pygame.K_3: action[0] = 3 # Blue
                if event.key == pygame.K_4: action[0] = 4 # White
                    
                # Fire
                if event.key == pygame.K_SPACE:
                    action[1] = 1

        if done:
            # If the game is over, wait for 'R' to reset
            pass
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()