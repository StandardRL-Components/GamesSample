import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
from typing import List, Dict, Tuple, Any
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Space to attack in the direction you last moved. "
        "Each action is one turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Command a robot in a grid-based arena to strategically defeat waves of "
        "increasingly challenging enemies in this turn-based tactical game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    CELL_SIZE = 40
    MAX_STEPS = 1000
    MAX_WAVES = 10
    ENEMIES_PER_WAVE = 7

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_ROBOT = (60, 150, 255)
    COLOR_ROBOT_GLOW = (60, 150, 255, 50)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_ENEMY_GLOW = (255, 80, 80, 50)
    COLOR_ATTACK = (255, 255, 100)
    COLOR_TEXT = (220, 220, 230)
    COLOR_HP_BAR_BG = (70, 70, 70)
    COLOR_HP_ROBOT = (60, 200, 60)
    COLOR_HP_ENEMY = (200, 60, 60)
    COLOR_GAMEOVER = (200, 0, 0, 180)
    COLOR_VICTORY = (0, 200, 0, 180)

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Game state variables
        self.steps: int = 0
        self.score: float = 0.0
        self.game_over: bool = False
        self.victory: bool = False
        self.wave_number: int = 0

        self.robot_pos: List[int] = [0, 0]
        self.robot_health: int = 0
        self.robot_max_health: int = 30
        self.last_move_direction: List[int] = [0, 1]  # Default facing down

        self.enemies: List[Dict[str, Any]] = []
        self.particles: List[Dict[str, Any]] = []

        # This will be properly initialized in reset()
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.wave_number = 1
        self.particles = []

        # Initialize robot
        self.robot_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.robot_health = self.robot_max_health
        self.last_move_direction = [0, 1]

        # Spawn first wave
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1

        # --- Player's Turn ---
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1

        if space_pressed:
            dmg_dealt, enemies_killed = self._handle_player_attack()
            reward += dmg_dealt * 0.1
            reward += enemies_killed * 5.0
        elif movement != 0:
            self._handle_player_movement(movement)
        # else: no-op

        # --- Enemy's Turn ---
        damage_taken = self._handle_enemy_actions()
        reward -= damage_taken * 0.2

        # --- State Update ---
        # Remove defeated enemies
        self.enemies = [e for e in self.enemies if e['health'] > 0]

        # Check for wave clear
        if not self.enemies:
            reward += 100.0
            self.wave_number += 1
            if self.wave_number > self.MAX_WAVES:
                self.victory = True
                self.game_over = True
            else:
                self._spawn_wave()
                self._create_text_particle(f"WAVE {self.wave_number}", (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2), self.font_large, self.COLOR_TEXT, 60)

        # Check for termination conditions
        terminated = False
        truncated = False
        if self.robot_health <= 0:
            reward -= 100.0
            self.game_over = True
            terminated = True
        
        if self.game_over:
            terminated = True

        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _spawn_wave(self):
        self.enemies.clear()
        enemy_health = min(10 + (self.wave_number - 1) * 5, 35)

        possible_spawns = list(
            (x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)
        )
        possible_spawns.remove(tuple(self.robot_pos))

        spawn_indices = self.np_random.choice(
            len(possible_spawns), size=min(self.ENEMIES_PER_WAVE, len(possible_spawns)), replace=False
        )
        
        for i in spawn_indices:
            pos = list(possible_spawns[i])
            self.enemies.append({
                "pos": pos,
                "health": enemy_health,
                "max_health": enemy_health
            })

    def _handle_player_attack(self) -> Tuple[int, int]:
        target_pos = [
            self.robot_pos[0] + self.last_move_direction[0],
            self.robot_pos[1] + self.last_move_direction[1]
        ]

        damage_dealt = 0
        enemies_killed = 0

        pixel_pos = self._grid_to_pixel_center(target_pos)
        self._create_flash_particle(pixel_pos, 30, self.COLOR_ATTACK, 10)

        for enemy in self.enemies:
            if enemy['pos'] == target_pos:
                damage = 10
                enemy['health'] -= damage
                damage_dealt += damage
                
                if enemy['health'] <= 0:
                    enemies_killed += 1
                
                text_pos = self._grid_to_pixel_center(enemy['pos'])
                self._create_text_particle(str(damage), (text_pos[0], text_pos[1]-20), self.font_medium, self.COLOR_ATTACK, 30, dy=-1)
                break
        
        return damage_dealt, enemies_killed

    def _handle_player_movement(self, movement_action: int):
        dx, dy = 0, 0
        if movement_action == 1: dy = -1  # Up
        elif movement_action == 2: dy = 1   # Down
        elif movement_action == 3: dx = -1  # Left
        elif movement_action == 4: dx = 1   # Right

        if dx == 0 and dy == 0:
            return

        new_pos = [self.robot_pos[0] + dx, self.robot_pos[1] + dy]

        if not (0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT):
            return

        if any(enemy['pos'] == new_pos for enemy in self.enemies):
            return

        self.robot_pos = new_pos
        self.last_move_direction = [dx, dy]

    def _handle_enemy_actions(self) -> int:
        total_damage_taken = 0
        all_entity_positions = {tuple(e['pos']) for e in self.enemies}
        all_entity_positions.add(tuple(self.robot_pos))

        for enemy in self.enemies:
            dist_x = self.robot_pos[0] - enemy['pos'][0]
            dist_y = self.robot_pos[1] - enemy['pos'][1]
            manhattan_dist = abs(dist_x) + abs(dist_y)

            if manhattan_dist == 1: # Adjacent, so attack
                damage = 5
                self.robot_health -= damage
                total_damage_taken += damage
                pixel_pos = self._grid_to_pixel_center(self.robot_pos)
                self._create_flash_particle(pixel_pos, 25, self.COLOR_ENEMY, 15)
                self._create_text_particle(str(damage), (pixel_pos[0], pixel_pos[1]-20), self.font_medium, self.COLOR_ENEMY, 30, dy=-1)

            elif manhattan_dist <= 3: # In range, move towards robot
                moves = []
                if dist_x != 0: moves.append((np.sign(dist_x), 0))
                if dist_y != 0: moves.append((0, np.sign(dist_y)))
                
                if moves:
                    if len(moves) > 1:
                        move_idx = self.np_random.integers(len(moves))
                        dx, dy = moves[move_idx]
                    else:
                        dx, dy = moves[0]

                    new_pos = (enemy['pos'][0] + int(dx), enemy['pos'][1] + int(dy))
                    if new_pos not in all_entity_positions:
                        old_pos = tuple(enemy['pos'])
                        enemy['pos'] = list(new_pos)
                        all_entity_positions.remove(old_pos)
                        all_entity_positions.add(new_pos)
            else: # Out of range, move randomly
                moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                self.np_random.shuffle(moves)
                for dx, dy in moves:
                    new_pos = (enemy['pos'][0] + dx, enemy['pos'][1] + dy)
                    if (0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT 
                        and new_pos not in all_entity_positions):
                        old_pos = tuple(enemy['pos'])
                        enemy['pos'] = list(new_pos)
                        all_entity_positions.remove(old_pos)
                        all_entity_positions.add(new_pos)
                        break
        
        self.robot_health = max(0, self.robot_health)
        return total_damage_taken

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_draw_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "robot_health": self.robot_health,
            "enemies_left": len(self.enemies)
        }

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

        # Draw attack indicator
        target_pos = [self.robot_pos[0] + self.last_move_direction[0], self.robot_pos[1] + self.last_move_direction[1]]
        if 0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT:
            px, py = self._grid_to_pixel_center(target_pos)
            pygame.draw.line(self.screen, self.COLOR_ATTACK, (px-5, py), (px+5, py), 1)
            pygame.draw.line(self.screen, self.COLOR_ATTACK, (px, py-5), (px, py+5), 1)

        # Draw enemies
        for enemy in self.enemies:
            px, py = self._grid_to_pixel_center(enemy['pos'])
            radius = self.CELL_SIZE // 2 - 5
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius + 3, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_ENEMY)
            self._draw_health_bar(self.screen, (px - 15, py - 25), (30, 5), enemy['health'], enemy['max_health'], self.COLOR_HP_ENEMY)

        # Draw robot
        px, py = self._grid_to_pixel_center(self.robot_pos)
        radius = self.CELL_SIZE // 2 - 4
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius + 4, self.COLOR_ROBOT_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, (px - radius, py - radius, radius*2, radius*2))
        
    def _render_ui(self):
        ui_bar_pos = (20, self.SCREEN_HEIGHT - 40)
        ui_bar_size = (200, 20)
        self._draw_health_bar(self.screen, ui_bar_pos, ui_bar_size, self.robot_health, self.robot_max_health, self.COLOR_HP_ROBOT)
        health_text = self.font_small.render(f"HP: {self.robot_health}/{self.robot_max_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (ui_bar_pos[0] + ui_bar_size[0] + 10, ui_bar_pos[1] + 2))

        wave_text = self.font_medium.render(f"WAVE: {self.wave_number}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (20, 10))

        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            if self.victory:
                overlay.fill(self.COLOR_VICTORY)
                end_text = self.font_large.render("VICTORY!", True, self.COLOR_TEXT)
            else:
                overlay.fill(self.COLOR_GAMEOVER)
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_TEXT)

            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(end_text, text_rect)

    def _draw_health_bar(self, surface, pos, size, current_hp, max_hp, color):
        x, y = pos
        w, h = size
        pygame.draw.rect(surface, self.COLOR_HP_BAR_BG, (x, y, w, h))
        fill_ratio = max(0, current_hp / max_hp) if max_hp > 0 else 0
        pygame.draw.rect(surface, color, (x, y, w * fill_ratio, h))
        pygame.draw.rect(surface, self.COLOR_TEXT, (x, y, w, h), 1)

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            
            if p['type'] == 'text':
                p['pos'] = (p['pos'][0] + p['dx'], p['pos'][1] + p['dy'])
                alpha = min(255, int(255 * (p['life'] / p['start_life'])))
                text_surf = p['font'].render(p['text'], True, p['color'])
                text_surf.set_alpha(alpha)
                text_rect = text_surf.get_rect(center=p['pos'])
                self.screen.blit(text_surf, text_rect)
            
            elif p['type'] == 'flash':
                alpha = int(150 * math.sin(math.pi * (p['life'] / p['start_life'])))
                color = (*p['color'], alpha)
                radius = int(p['radius'] * (1.0 - (p['life'] / p['start_life'])))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def _create_text_particle(self, text, pos, font, color, life, dx=0, dy=0):
        self.particles.append({
            'type': 'text', 'text': text, 'pos': list(pos), 'font': font, 'color': color,
            'life': life, 'start_life': life, 'dx': dx, 'dy': dy
        })

    def _create_flash_particle(self, pos, radius, color, life):
        self.particles.append({
            'type': 'flash', 'pos': pos, 'radius': radius, 'color': color,
            'life': life, 'start_life': life
        })

    def _grid_to_pixel_center(self, grid_pos: List[int]) -> Tuple[int, int]:
        px = int(grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        py = int(grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        return px, py

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in the headless environment.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    running = True
    game_terminated = False
    
    pygame.display.set_caption("Grid Robot Arena")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(GameEnv.user_guide)
    print("Controls: R to reset.")
    print("="*30 + "\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not game_terminated:
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    game_terminated = False
                    print("--- Game Reset ---")
                    continue
                
                one_time_action = [0, 0, 0]
                if event.key == pygame.K_UP: one_time_action[0] = 1
                elif event.key == pygame.K_DOWN: one_time_action[0] = 2
                elif event.key == pygame.K_LEFT: one_time_action[0] = 3
                elif event.key == pygame.K_RIGHT: one_time_action[0] = 4
                elif event.key == pygame.K_SPACE: one_time_action[1] = 1
                
                if any(one_time_action):
                    obs, reward, terminated, truncated, info = env.step(one_time_action)
                    game_terminated = terminated or truncated
                    print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}, Truncated: {truncated}")
                    if game_terminated:
                        print(f"Final Info: {info}")

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(10)

    env.close()