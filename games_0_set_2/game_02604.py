# Generated: 2025-08-27T20:52:17.337913
# Source Brief: brief_02604.md
# Brief Index: 2604

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your character (the white square). "
        "Reach the village (yellow) before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape a haunted forest before the lurking horrors (red) catch you. "
        "Each move counts as the forest grows more dangerous."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // TILE_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // TILE_SIZE

    TIME_LIMIT = 180
    MAX_ENCOUNTERS = 4
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_TREE = (20, 35, 55)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255, 30)
    COLOR_ENEMY = (220, 40, 40)
    COLOR_ENEMY_GLOW = (255, 100, 100, 60)
    COLOR_VILLAGE = (200, 200, 150)
    COLOR_VILLAGE_GLOW = (255, 255, 200, 50)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_UI_DANGER = (255, 80, 80)
    COLOR_PARTICLE_BLOOD = (180, 20, 20)

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
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)

        self._initialize_surfaces()
        
        # self.reset() is called here, but its attributes must be initialized first
        # We will call it after all members are initialized, or handle initialization within reset itself.
        # The original code called reset() here which caused the error.
        # We'll let the user/agent call reset() as per standard Gym API.
        # For the validation, we'll manually call it.
        # However, to maintain the original code's behavior of being ready after __init__,
        # we will call reset() at the end of __init__.
        self.reset()
        
        # self.validate_implementation() # This is a helper, not part of the core env

    def _initialize_surfaces(self):
        """Create pre-rendered surfaces for performance and visual effects."""
        self.fog_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        self.fog_surface.fill((50, 60, 80, 0))
        for _ in range(150):
            pos = (random.randint(-100, self.SCREEN_WIDTH + 100), random.randint(-100, self.SCREEN_HEIGHT + 100))
            radius = random.randint(50, 150)
            alpha = random.randint(15, 35)
            pygame.gfxdraw.filled_circle(self.fog_surface, pos[0], pos[1], radius, (50, 60, 80, alpha))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.encounter_count = 0

        # --- Player and World ---
        self.player_pos = (1, self.GRID_HEIGHT // 2)
        self.village_pos = (self.GRID_WIDTH - 2, self.np_random.integers(1, self.GRID_HEIGHT - 1))

        # --- Procedural Generation ---
        self.trees = self._generate_trees()
        
        # --- Difficulty Scaling ---
        self.enemy_spawn_chance = 0.01
        self.enemy_base_speed = 1.0

        # --- Enemies ---
        self.enemies = []
        self._spawn_initial_enemy()
        
        # --- Effects ---
        self.particles = []

        return self._get_observation(), self._get_info()

    def _generate_trees(self):
        trees = []
        for _ in range(int(self.GRID_WIDTH * self.GRID_HEIGHT * 0.2)):
            pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
            if pos != self.player_pos and pos != self.village_pos:
                trees.append(pos)
        return trees

    def _spawn_initial_enemy(self):
        while True:
            pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
            dist = abs(pos[0] - self.player_pos[0]) + abs(pos[1] - self.player_pos[1])
            if dist > 5 and pos not in self.trees:
                self.enemies.append({
                    'pos': pos,
                    'speed': self.enemy_base_speed,
                    'movement_accumulator': 0.0
                })
                break
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        
        # --- Player Movement and Reward ---
        dist_before = self._manhattan_distance(self.player_pos, self.village_pos)
        self._move_player(movement)
        dist_after = self._manhattan_distance(self.player_pos, self.village_pos)

        if dist_after < dist_before:
            reward += 0.1
        elif dist_after > dist_before:
            reward -= 0.2
            
        # --- Enemy Logic ---
        self._update_enemies()
        
        # --- Collision and Encounters ---
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy['pos'] == self.player_pos:
                self.encounter_count += 1
                reward -= 5.0
                enemies_to_remove.append(enemy)
                self._create_encounter_particles(self.player_pos)
                # // SFX: Ghostly wail

        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        
        # --- Game Progression ---
        self._update_difficulty()
        self._try_spawn_enemy()
        self._update_particles()
        
        self.steps += 1
        
        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated:
            if self.player_pos == self.village_pos:
                reward += 100.0
                self.game_over_message = "YOU ESCAPED"
                # // SFX: Village bells, sigh of relief
            elif self.encounter_count >= self.MAX_ENCOUNTERS:
                reward -= 100.0
                self.game_over_message = "THEY GOT YOU"
                # // SFX: Horror sting, final scream
            elif self.steps >= self.TIME_LIMIT:
                self.game_over_message = "LOST IN THE DARK"
                # // SFX: Ominous wind
        
        self.score += reward

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        terminated = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _move_player(self, movement):
        px, py = self.player_pos
        if movement == 1: py -= 1  # Up
        elif movement == 2: py += 1  # Down
        elif movement == 3: px -= 1  # Left
        elif movement == 4: px += 1  # Right
        
        px = max(0, min(self.GRID_WIDTH - 1, px))
        py = max(0, min(self.GRID_HEIGHT - 1, py))
        
        if (px, py) not in self.trees:
            self.player_pos = (px, py)

    def _update_enemies(self):
        for enemy in self.enemies:
            dist_to_player = self._manhattan_distance(enemy['pos'], self.player_pos)
            
            if dist_to_player <= 3:
                enemy['movement_accumulator'] += enemy['speed']
                moves_to_make = int(enemy['movement_accumulator'])
                if moves_to_make > 0:
                    enemy['movement_accumulator'] -= moves_to_make
                    
                    ex, ey = enemy['pos']
                    for _ in range(moves_to_make):
                        px, py = self.player_pos
                        dx, dy = px - ex, py - ey
                        
                        if abs(dx) > abs(dy):
                            ex += int(np.sign(dx))
                        elif abs(dy) > 0:
                            ey += int(np.sign(dy))
                        
                        # Enemies can't move into trees
                        if (ex, ey) not in self.trees:
                            enemy['pos'] = (ex, ey)
                        else: # a tree is blocking, stop moving
                            ex, ey = enemy['pos']
                            break

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 50 == 0:
            self.enemy_spawn_chance += 0.01
        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_base_speed += 0.1
            for e in self.enemies:
                e['speed'] = self.enemy_base_speed

    def _try_spawn_enemy(self):
        if self.np_random.random() < self.enemy_spawn_chance:
            for _ in range(10): # Try 10 times to find a valid spot
                spawn_dist = max(4, 15 - self.steps // 20)
                angle = self.np_random.uniform(0, 2 * math.pi)
                x = int(self.player_pos[0] + math.cos(angle) * spawn_dist)
                y = int(self.player_pos[1] + math.sin(angle) * spawn_dist)

                x = max(0, min(self.GRID_WIDTH - 1, x))
                y = max(0, min(self.GRID_HEIGHT - 1, y))

                pos = (x, y)
                if pos not in self.trees and pos != self.player_pos and pos != self.village_pos:
                    self.enemies.append({
                        'pos': pos,
                        'speed': self.enemy_base_speed,
                        'movement_accumulator': 0.0
                    })
                    break

    def _check_termination(self):
        terminated = False
        if self.player_pos == self.village_pos:
            self.game_over = True
            terminated = True
        elif self.encounter_count >= self.MAX_ENCOUNTERS:
            self.game_over = True
            terminated = True
        elif self.steps >= self.TIME_LIMIT:
            self.game_over = True
            terminated = True
        return terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self.screen.blit(self.fog_surface, (0, 0))
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_trees()
        self._render_village()
        self._render_enemies()
        self._render_player()
        self._render_particles()

    def _grid_to_pixel_center(self, grid_pos):
        x = grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2
        y = grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        return int(x), int(y)

    def _render_trees(self):
        for pos in self.trees:
            px, py = self._grid_to_pixel_center(pos)
            radius = self.TILE_SIZE // 2
            pygame.draw.rect(self.screen, self.COLOR_TREE, (px - radius, py - radius, self.TILE_SIZE, self.TILE_SIZE))

    def _render_village(self):
        px, py = self._grid_to_pixel_center(self.village_pos)
        glow_radius = int(self.TILE_SIZE * 1.5)
        pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, self.COLOR_VILLAGE_GLOW)
        pygame.gfxdraw.aacircle(self.screen, px, py, glow_radius, self.COLOR_VILLAGE_GLOW)
        
        rect = pygame.Rect(px - self.TILE_SIZE//2, py - self.TILE_SIZE//2, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_VILLAGE, rect)

    def _render_enemies(self):
        for enemy in self.enemies:
            px, py = self._grid_to_pixel_center(enemy['pos'])
            
            # Glow
            glow_radius = int(self.TILE_SIZE * (0.8 + 0.2 * math.sin(self.steps * 0.5 + id(enemy))))
            pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.aacircle(self.screen, px, py, glow_radius, self.COLOR_ENEMY_GLOW)

            # Core
            flicker = 0.8 + 0.2 * math.sin(self.steps * 0.5 + id(enemy))
            size = int(self.TILE_SIZE * 0.7 * flicker)
            rect = pygame.Rect(px - size//2, py - size//2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect)
            
    def _render_player(self):
        px, py = self._grid_to_pixel_center(self.player_pos)
        
        # Glow
        glow_radius = int(self.TILE_SIZE * 1.2)
        pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, px, py, glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Core
        size = self.TILE_SIZE - 4
        rect = pygame.Rect(px - size//2, py - size//2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = p['life'] / p['max_life']
            color = (*self.COLOR_PARTICLE_BLOOD, int(255 * alpha))
            radius = int(p['radius'] * alpha)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _create_encounter_particles(self, grid_pos):
        px, py = self._grid_to_pixel_center(grid_pos)
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.integers(3, 7),
                'life': self.np_random.integers(15, 30),
                'max_life': 30
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['life'] -= 1

    def _render_ui(self):
        # Encounters
        encounter_text = f"ENCOUNTERS: {self.encounter_count} / {self.MAX_ENCOUNTERS}"
        text_surface = self.font_medium.render(encounter_text, True, self.COLOR_UI_DANGER)
        self.screen.blit(text_surface, (10, 10))
        
        # Timer
        time_left = max(0, self.TIME_LIMIT - self.steps)
        timer_text = f"TIME: {time_left}"
        text_surface = self.font_medium.render(timer_text, True, self.COLOR_UI_TEXT)
        text_rect = text_surface.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surface, text_rect)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message_surface = self.font_large.render(self.game_over_message, True, self.COLOR_UI_TEXT)
            message_rect = message_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(message_surface, message_rect)

            score_surface = self.font_medium.render(f"Final Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
            score_rect = score_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30))
            self.screen.blit(score_surface, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "encounter_count": self.encounter_count,
            "player_pos": self.player_pos,
            "village_pos": self.village_pos,
        }

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is important to unset the dummy video driver for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Haunted Forest Escape")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    print("--- Human Play Mode ---")
    print(GameEnv.user_guide)
    print("-----------------------")

    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        # Event handling
        manual_step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                elif event.key == pygame.K_q:
                    running = False
                else:
                    manual_step = True
        
        if not env.game_over:
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            # In manual play, we only step if a key is pressed (turn-based)
            if manual_step:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
                if terminated or truncated:
                    print(f"Game Over! Final Score: {info['score']:.2f}")
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        # Pygame uses (width, height), but our obs is (height, width, 3)
        # So we need to transpose it back for display
        frame_to_show = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame_to_show)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate for human play
        
    env.close()