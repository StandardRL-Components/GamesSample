
# Generated: 2025-08-27T20:00:59.599689
# Source Brief: brief_02324.md
# Brief Index: 2324

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move one square. Press Space to attack all "
        "adjacent squares. You can move and attack in the same turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A tactical, grid-based strategy game. Command your robot to defeat waves "
        "of enemies by carefully choosing your position and when to attack."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.CELL_WIDTH = self.SCREEN_WIDTH // self.GRID_WIDTH
        self.CELL_HEIGHT = self.SCREEN_HEIGHT // self.GRID_HEIGHT
        self.MAX_STEPS = 1000
        
        self.ROBOT_MAX_HEALTH = 10
        self.ROBOT_ATTACK_POWER = 2
        
        # --- Colors ---
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_GRID = (45, 45, 60)
        self.COLOR_ROBOT = (60, 180, 255)
        self.COLOR_ROBOT_GLOW = (120, 210, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BG = (80, 20, 20)
        self.COLOR_HEALTH_FG = (50, 220, 50)
        self.ENEMY_COLORS = {
            "chaser": (255, 80, 80),
            "brute": (200, 100, 255),
            "scout": (255, 255, 100),
        }

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.robot_pos = [0, 0]
        self.robot_health = 0
        self.enemies = []
        self.visual_effects = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 1
        
        self.robot_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.robot_health = self.ROBOT_MAX_HEALTH
        
        self.enemies = []
        self.visual_effects = []
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1
        self._update_effects()

        # --- Player Turn ---
        movement = action[0]
        is_attacking = action[1] == 1
        
        # 1. Player Movement
        if movement > 0:
            target_pos = self.robot_pos[:]
            if movement == 1: target_pos[1] -= 1  # Up
            elif movement == 2: target_pos[1] += 1  # Down
            elif movement == 3: target_pos[0] -= 1  # Left
            elif movement == 4: target_pos[0] += 1  # Right
            
            if 0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT:
                self.robot_pos = target_pos

        # 2. Player Attack
        if is_attacking:
            # Sound: Player_Attack.wav
            attack_range = self._get_adjacent_tiles(self.robot_pos)
            self._add_effect('attack_blast', self.robot_pos, duration=1)
            
            dead_enemies = []
            for enemy in self.enemies:
                if tuple(enemy['pos']) in attack_range:
                    enemy['health'] -= self.ROBOT_ATTACK_POWER
                    reward += 0.1
                    self._add_effect('hit_spark', enemy['pos'], duration=1)
                    if enemy['health'] <= 0:
                        dead_enemies.append(enemy)

            for dead_enemy in dead_enemies:
                # Sound: Enemy_Explosion.wav
                self.enemies.remove(dead_enemy)
                self.score += 100
                reward += 1.0
                self._add_effect('explosion', dead_enemy['pos'], duration=2)
        
        # --- Enemy Turn ---
        enemy_positions = {tuple(e['pos']) for e in self.enemies}
        for enemy in self.enemies:
            if self._manhattan_distance(enemy['pos'], self.robot_pos) == 1:
                # Attack
                # Sound: Player_Hit.wav
                self.robot_health -= enemy['attack']
                reward -= 0.1
                self._add_effect('player_hit', self.robot_pos, duration=1)
            else:
                # Move
                self._move_enemy(enemy, enemy_positions)
        
        # --- Check Game State ---
        terminated = False
        if self.robot_health <= 0:
            # Sound: Game_Over.wav
            self.robot_health = 0
            self.game_over = True
            terminated = True
            reward -= 10.0
        
        if not self.enemies and not self.game_over:
            # Sound: Wave_Clear.wav
            self.wave_number += 1
            self.score += 500
            reward += 10.0
            self._spawn_wave()

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number}

    def _spawn_wave(self):
        num_enemies = self.wave_number + 1
        occupied_tiles = {tuple(self.robot_pos)}
        
        for _ in range(num_enemies):
            spawn_pos = None
            for _ in range(100): # Failsafe against infinite loop
                pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]
                if tuple(pos) not in occupied_tiles and self._manhattan_distance(pos, self.robot_pos) > 3:
                    spawn_pos = pos
                    break
            if spawn_pos is None: continue # Skip if no space found

            occupied_tiles.add(tuple(spawn_pos))
            
            enemy_type = self.np_random.choice(["chaser", "brute", "scout"])
            health, attack = 2, 1
            if enemy_type == "brute": health, attack = 4, 1
            if enemy_type == "scout": health, attack = 1, 2
            
            if self.np_random.random() < 0.25 * (self.wave_number - 1):
                health += 1

            self.enemies.append({
                "pos": spawn_pos,
                "type": enemy_type,
                "health": health,
                "max_health": health,
                "attack": attack
            })
            
    def _move_enemy(self, enemy, occupied_tiles):
        if enemy['type'] == 'chaser' or enemy['type'] == 'brute':
            # Move towards player
            dx = self.robot_pos[0] - enemy['pos'][0]
            dy = self.robot_pos[1] - enemy['pos'][1]
            
            potential_moves = []
            if dx != 0: potential_moves.append([np.sign(dx), 0])
            if dy != 0: potential_moves.append([0, np.sign(dy)])
            self.np_random.shuffle(potential_moves)
            
            for move in potential_moves:
                next_pos = [enemy['pos'][0] + move[0], enemy['pos'][1] + move[1]]
                if tuple(next_pos) not in occupied_tiles:
                    occupied_tiles.remove(tuple(enemy['pos']))
                    enemy['pos'] = next_pos
                    occupied_tiles.add(tuple(enemy['pos']))
                    return
        elif enemy['type'] == 'scout':
            # Move randomly
            moves = [[0,1], [0,-1], [1,0], [-1,0]]
            self.np_random.shuffle(moves)
            for move in moves:
                next_pos = [enemy['pos'][0] + move[0], enemy['pos'][1] + move[1]]
                if 0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT and tuple(next_pos) not in occupied_tiles:
                    occupied_tiles.remove(tuple(enemy['pos']))
                    enemy['pos'] = next_pos
                    occupied_tiles.add(tuple(enemy['pos']))
                    return

    def _grid_to_pixel(self, grid_pos):
        gx, gy = grid_pos
        px = int(gx * self.CELL_WIDTH + self.CELL_WIDTH * 0.5)
        py = int(gy * self.CELL_HEIGHT + self.CELL_HEIGHT * 0.5)
        return px, py

    def _render_game(self):
        self._render_grid()
        for enemy in self.enemies:
            self._render_entity(enemy, self.ENEMY_COLORS[enemy['type']])
        self._render_entity({**self.__dict__, 'pos': self.robot_pos, 'health': self.robot_health, 'max_health': self.ROBOT_MAX_HEALTH}, self.COLOR_ROBOT, glow=True)
        self._render_effects()

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (x * self.CELL_WIDTH, 0)
            end_pos = (x * self.CELL_WIDTH, self.SCREEN_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (0, y * self.CELL_HEIGHT)
            end_pos = (self.SCREEN_WIDTH, y * self.CELL_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_entity(self, entity, color, glow=False):
        pos_px = self._grid_to_pixel(entity['pos'])
        radius = int(min(self.CELL_WIDTH, self.CELL_HEIGHT) * 0.35)
        
        if glow:
            glow_radius = int(radius * 1.5)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_ROBOT_GLOW, 50))
            self.screen.blit(glow_surf, (pos_px[0] - glow_radius, pos_px[1] - glow_radius))

        pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, color)
        pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], radius, color)
        
        self._render_health_bar(pos_px, radius, entity['health'], entity['max_health'])

    def _render_health_bar(self, pos_px, radius, current_hp, max_hp):
        if current_hp == max_hp: return
        bar_width = int(radius * 1.8)
        bar_height = 5
        y_offset = radius + 5
        
        bg_rect = pygame.Rect(pos_px[0] - bar_width // 2, pos_px[1] - y_offset, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect)
        
        health_ratio = max(0, current_hp / max_hp)
        fg_width = int(bar_width * health_ratio)
        fg_rect = pygame.Rect(pos_px[0] - bar_width // 2, pos_px[1] - y_offset, fg_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, fg_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, bg_rect, 1)

    def _render_effects(self):
        for effect in self.visual_effects:
            pos_px = self._grid_to_pixel(effect['pos'])
            progress = effect['timer'] / effect['duration']

            if effect['type'] == 'explosion':
                radius = int(self.CELL_WIDTH * (1.0 - progress))
                alpha = int(255 * progress)
                pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], radius, (*self.ENEMY_COLORS['chaser'], alpha))
                pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, (*(255,255,255), alpha))
            elif effect['type'] == 'attack_blast':
                radius = int(self.CELL_WIDTH * 1.5 * (1.0 - progress))
                alpha = int(150 * progress)
                if alpha > 0:
                    pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, (*(255,255,100), alpha))
            elif effect['type'] == 'hit_spark':
                pygame.draw.line(self.screen, (255,255,255), (pos_px[0]-5, pos_px[1]), (pos_px[0]+5, pos_px[1]), 2)
                pygame.draw.line(self.screen, (255,255,255), (pos_px[0], pos_px[1]-5), (pos_px[0], pos_px[1]+5), 2)
            elif effect['type'] == 'player_hit':
                s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                s.fill((255, 0, 0, int(100 * progress)))
                self.screen.blit(s, (0,0))
    
    def _render_ui(self):
        texts = [
            f"Wave: {self.wave_number}",
            f"Score: {self.score}",
            f"Health: {self.robot_health}/{self.ROBOT_MAX_HEALTH}"
        ]
        for i, text in enumerate(texts):
            text_surface = self.font_ui.render(text, True, self.COLOR_TEXT)
            self.screen.blit(text_surface, (10, 10 + i * 25))
        
        if self.game_over:
            text_surface = self.font_game_over.render("GAME OVER", True, self.ENEMY_COLORS['chaser'])
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surface, text_rect)

    def _add_effect(self, type, pos, duration):
        self.visual_effects.append({'type': type, 'pos': pos, 'duration': duration, 'timer': duration})

    def _update_effects(self):
        self.visual_effects = [e for e in self.visual_effects if e['timer'] > 0]
        for e in self.visual_effects: e['timer'] -= 1

    @staticmethod
    def _manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
    def _get_adjacent_tiles(self, pos):
        px, py = pos
        return {
            (px, py-1), (px, py+1), (px-1, py), (px+1, py),
            (px-1, py-1), (px+1, py-1), (px-1, py+1), (px+1, py+1)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Grid Combat Arena")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Event Handling ---
        action = [0, 0, 0] # Default action: no-op
        needs_step = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not terminated:
                needs_step = True
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()
                    terminated = False
                    needs_step = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # --- Game Step ---
        if needs_step and not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # --- Rendering ---
        # Convert the observation (H, W, C) to a Pygame surface (W, H)
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for human play

    env.close()