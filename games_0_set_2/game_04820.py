
# Generated: 2025-08-28T03:06:15.521759
# Source Brief: brief_04820.md
# Brief Index: 4820

        
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
        "Controls: Arrow keys to move one square. Space to attack the nearest enemy. You cannot move and attack in the same turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a robot in a grid-based arena. Defeat all enemies in turn-based combat to win. Evade enemy attacks and manage your health."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_STEPS = 1000
        self.NUM_ENEMIES = 3

        # Visual constants
        self.GRID_AREA_SIZE = 360
        self.CELL_SIZE = self.GRID_AREA_SIZE // self.GRID_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_AREA_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_AREA_SIZE) // 2
        
        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (30, 75, 128)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (128, 25, 25)
        self.COLOR_HEALTH_BAR_BG = (60, 60, 60)
        self.COLOR_HEALTH_FILL = (80, 220, 80)
        self.COLOR_ATTACK = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_DAMAGE_PLAYER = (255, 100, 100)
        self.COLOR_DAMAGE_ENEMY = (255, 255, 200)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_big = pygame.font.Font(None, 72)
        
        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.robot_pos = (0, 0)
        self.robot_health = 0
        self.max_robot_health = 100
        self.enemies = []
        self.max_enemy_health = 50
        self.visual_effects = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.visual_effects = []
        
        self.robot_health = self.max_robot_health
        
        # Generate all possible grid positions
        all_pos = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(all_pos)
        
        # Place robot
        self.robot_pos = all_pos.pop()
        
        # Place enemies
        self.enemies = []
        for _ in range(self.NUM_ENEMIES):
            if not all_pos: break
            enemy_pos = all_pos.pop()
            self.enemies.append({
                "pos": enemy_pos,
                "health": self.max_enemy_health
            })
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.1  # Cost of existing
        self.visual_effects = []

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        # --- Player's Turn ---
        action_taken = False
        # 1. Handle Movement (takes priority over attack)
        if movement > 0:
            action_taken = True
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            new_pos = (self.robot_pos[0] + dx, self.robot_pos[1] + dy)
            
            # Check boundaries and collisions
            if (0 <= new_pos[0] < self.GRID_SIZE and
                0 <= new_pos[1] < self.GRID_SIZE and
                new_pos not in [e['pos'] for e in self.enemies]):
                self.robot_pos = new_pos
        
        # 2. Handle Attack (if no movement was performed)
        elif space_pressed and not action_taken:
            action_taken = True
            if self.enemies:
                # Find nearest enemy/enemies
                min_dist = float('inf')
                nearest_enemies = []
                for i, enemy in enumerate(self.enemies):
                    dist = math.hypot(self.robot_pos[0] - enemy['pos'][0], self.robot_pos[1] - enemy['pos'][1])
                    if dist < min_dist:
                        min_dist = dist
                        nearest_enemies = [i]
                    elif dist == min_dist:
                        nearest_enemies.append(i)
                
                # Attack a random one from the nearest
                target_idx = self.np_random.choice(nearest_enemies)
                target_enemy = self.enemies[target_idx]
                
                damage = 20
                target_enemy['health'] -= damage
                reward += 1.0 # Reward for damaging
                
                # sfx: player_attack
                self._add_attack_effect(self.robot_pos, target_enemy['pos'], self.COLOR_ATTACK)
                self._add_damage_number_effect(target_enemy['pos'], str(damage), self.COLOR_DAMAGE_ENEMY)

                if target_enemy['health'] <= 0:
                    # sfx: enemy_destroyed
                    self.score += 50
                    self.enemies.pop(target_idx)

        # --- Enemy's Turn ---
        if action_taken:
            adjacent_enemies = []
            for i, enemy in enumerate(self.enemies):
                dist = math.hypot(self.robot_pos[0] - enemy['pos'][0], self.robot_pos[1] - enemy['pos'][1])
                if dist <= 1.01: # Adjacency check (including diagonals for simplicity)
                    adjacent_enemies.append(i)

            # One random adjacent enemy attacks
            if adjacent_enemies:
                attacker_idx = self.np_random.choice(adjacent_enemies)
                attacker = self.enemies[attacker_idx]
                
                damage = 10
                self.robot_health -= damage
                reward -= 2.0 # Penalty for taking damage
                
                # sfx: player_hit
                self._add_attack_effect(attacker['pos'], self.robot_pos, self.COLOR_ENEMY)
                self._add_damage_number_effect(self.robot_pos, str(damage), self.COLOR_DAMAGE_PLAYER)
                
                # Other enemies (non-adjacent) move
                for i, enemy in enumerate(self.enemies):
                    if i != attacker_idx and i not in adjacent_enemies:
                        self._move_enemy(i)
            else:
                # All enemies move
                for i in range(len(self.enemies)):
                    self._move_enemy(i)
        
        # --- Termination Check ---
        terminated = False
        if self.robot_health <= 0:
            self.robot_health = 0
            reward -= 100
            terminated = True
            self.game_over = True
            self.win_message = "YOU DIED"
        elif not self.enemies:
            reward += 100
            terminated = True
            self.game_over = True
            self.win_message = "VICTORY!"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_message = "TIME UP"
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_enemy(self, enemy_idx):
        enemy = self.enemies[enemy_idx]
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.np_random.shuffle(moves)

        for dx, dy in moves:
            new_pos = (enemy['pos'][0] + dx, enemy['pos'][1] + dy)
            if (0 <= new_pos[0] < self.GRID_SIZE and
                0 <= new_pos[1] < self.GRID_SIZE and
                new_pos != self.robot_pos and
                new_pos not in [e['pos'] for e in self.enemies]):
                enemy['pos'] = new_pos
                break

    def _grid_to_pixel(self, grid_pos):
        px = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _add_attack_effect(self, start_grid_pos, end_grid_pos, color):
        self.visual_effects.append({
            "type": "attack_line",
            "start": self._grid_to_pixel(start_grid_pos),
            "end": self._grid_to_pixel(end_grid_pos),
            "color": color
        })

    def _add_damage_number_effect(self, grid_pos, text, color):
        pixel_pos = self._grid_to_pixel(grid_pos)
        self.visual_effects.append({
            "type": "damage_number",
            "pos": (pixel_pos[0], pixel_pos[1] - self.CELL_SIZE // 2),
            "text": text,
            "color": color
        })
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_v = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_v = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_AREA_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_v, end_v)
            # Horizontal
            start_h = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_h = (self.GRID_OFFSET_X + self.GRID_AREA_SIZE, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_h, end_h)

        # Draw enemies
        for enemy in self.enemies:
            center_px = self._grid_to_pixel(enemy['pos'])
            size = int(self.CELL_SIZE * 0.8)
            glow_size = int(size * 1.2)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_GLOW, (center_px[0] - glow_size // 2, center_px[1] - glow_size // 2, glow_size, glow_size))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (center_px[0] - size // 2, center_px[1] - size // 2, size, size))
            
            # Enemy health bar
            bar_w = int(self.CELL_SIZE * 0.8)
            bar_h = 4
            bar_y = center_px[1] + size // 2 + 2
            fill_w = int(bar_w * (enemy['health'] / self.max_enemy_health))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (center_px[0] - bar_w // 2, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (center_px[0] - bar_w // 2, bar_y, fill_w, bar_h))


        # Draw player
        center_px = self._grid_to_pixel(self.robot_pos)
        size = int(self.CELL_SIZE * 0.8)
        glow_size = int(size * 1.4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, (center_px[0] - glow_size // 2, center_px[1] - glow_size // 2, glow_size, glow_size))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (center_px[0] - size // 2, center_px[1] - size // 2, size, size))

        # Draw visual effects
        for effect in self.visual_effects:
            if effect["type"] == "attack_line":
                pygame.draw.line(self.screen, effect["color"], effect["start"], effect["end"], 3)
            elif effect["type"] == "damage_number":
                text_surf = self.font_ui.render(effect["text"], True, effect["color"])
                text_rect = text_surf.get_rect(center=effect["pos"])
                self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Health Bar
        bar_width = 200
        bar_height = 20
        health_fill_width = int(bar_width * (self.robot_health / self.max_robot_health))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (20, 20, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FILL, (20, 20, health_fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (20, 20, bar_width, bar_height), 1)
        
        health_text = self.font_ui.render("HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (25, 22))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_big.render(self.win_message, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "robot_health": self.robot_health,
            "enemies_left": len(self.enemies),
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's a demonstration of how the environment works
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Robot Arena")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")
    
    running = True
    terminated = False
    
    while running:
        # Get action from keyboard
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    terminated = False
                    
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            
            # Since auto_advance is False, we only step on an action.
            # For human play, we need to decide when to send an action.
            # Let's send an action on any key press to make it feel turn-based.
            # A simple way is to check if any relevant key is down, but that would
            # step every frame. We need to wait for a key *press*.
            # The current event loop is not set up for that, so for this demo,
            # we will step continuously if a key is held.
            
            # To make it playable, we only step if an action is intended
            if any(action):
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")
            
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Limit frame rate
        clock.tick(10) # Low FPS for turn-based feel
        
    env.close()