
# Generated: 2025-08-27T15:26:29.759024
# Source Brief: brief_00994.md
# Brief Index: 994

        
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
    """
    A Gymnasium environment for a grid-based robot combat game.

    The player controls a robot on an 8x8 grid, fighting against 7 enemies.
    The game is turn-based: the player acts, then all enemies act.
    The goal is to defeat all enemies before the robot's health reaches zero.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]` (Movement): 0=None, 1=Up, 2=Down, 3=Left, 4=Right
    - `action[1]` (Space): 1=Attack nearest enemy in range
    - `action[2]` (Shift): 1=Attack lowest-health enemy in range

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    A 640x400 RGB image of the game state.

    **Rewards:**
    - -0.1 per step (encourages efficiency)
    - +1.0 for damaging an enemy
    - -1.0 for taking damage
    - +100 for winning (defeating all enemies)
    - -100 for losing (robot health <= 0)
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Space to attack the NEAREST enemy in range. "
        "Shift to attack the WEAKEST enemy in range."
    )

    game_description = (
        "Pilot a robot in a grid-based arena. Strategically move and attack to "
        "defeat all 7 enemies and claim victory."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 8
    CELL_SIZE = 40
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_SIZE * CELL_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_SIZE * CELL_SIZE) // 2

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (50, 60, 70)
    COLOR_ROBOT = (0, 255, 128)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_HEALTH_BG = (100, 0, 0)
    COLOR_HEALTH_FG = (0, 200, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_DAMAGE = (255, 200, 0)

    # Game parameters
    PLAYER_MAX_HEALTH = 10
    ENEMY_MAX_HEALTH = 3
    NUM_ENEMIES = 7
    PLAYER_ATTACK_RANGE = 1
    PLAYER_ATTACK_DAMAGE = 1
    ENEMY_ATTACK_RANGE = 1
    ENEMY_ATTACK_DAMAGE = 1
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables are initialized in reset()
        self.robot_pos = None
        self.robot_health = None
        self.enemies = []
        self.damage_particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.damage_particles = []

        # Place robot in the center
        self.robot_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        self.robot_health = self.PLAYER_MAX_HEALTH

        # Place enemies randomly
        self.enemies = []
        occupied_positions = {tuple(self.robot_pos)}
        for _ in range(self.NUM_ENEMIES):
            while True:
                pos = self.np_random.integers(0, self.GRID_SIZE, size=2)
                if tuple(pos) not in occupied_positions:
                    occupied_positions.add(tuple(pos))
                    self.enemies.append({
                        "pos": pos,
                        "health": self.ENEMY_MAX_HEALTH,
                        "alive": True,
                        "id": len(self.enemies)
                    })
                    break
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.1  # Time penalty
        self.steps += 1
        
        # --- Player Turn ---
        reward += self._handle_player_action(action)

        # --- Enemy Turn ---
        if not self._check_win_condition():
            reward += self._handle_enemy_turn()

        # --- Update & Check Termination ---
        self._update_particles()
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        action_reward = 0

        # Attacks take priority over movement
        attack_performed = False
        if shift_held or space_held:
            targets_in_range = []
            for enemy in self.enemies:
                if enemy["alive"]:
                    dist = np.sum(np.abs(self.robot_pos - enemy["pos"]))
                    if dist <= self.PLAYER_ATTACK_RANGE:
                        targets_in_range.append(enemy)

            if targets_in_range:
                target_enemy = None
                if shift_held: # Prioritize shift: attack weakest
                    target_enemy = min(targets_in_range, key=lambda e: e["health"])
                elif space_held: # Attack nearest
                    target_enemy = min(targets_in_range, key=lambda e: np.sum(np.abs(self.robot_pos - e["pos"])))
                
                if target_enemy:
                    # // SFX: Player laser shot
                    target_enemy["health"] -= self.PLAYER_ATTACK_DAMAGE
                    action_reward += 1.0
                    self.score += 10
                    self._create_damage_particle(target_enemy["pos"], self.PLAYER_ATTACK_DAMAGE)
                    if target_enemy["health"] <= 0:
                        target_enemy["alive"] = False
                        # // SFX: Enemy explosion
                        self.score += 50
                    attack_performed = True

        if not attack_performed and movement != 0:
            # // SFX: Robot move
            new_pos = self.robot_pos.copy()
            if movement == 1: new_pos[1] -= 1  # Up
            elif movement == 2: new_pos[1] += 1  # Down
            elif movement == 3: new_pos[0] -= 1  # Left
            elif movement == 4: new_pos[0] += 1  # Right
            
            # Clamp to grid boundaries
            new_pos[0] = np.clip(new_pos[0], 0, self.GRID_SIZE - 1)
            new_pos[1] = np.clip(new_pos[1], 0, self.GRID_SIZE - 1)
            self.robot_pos = new_pos
        
        return action_reward

    def _handle_enemy_turn(self):
        damage_taken_reward = 0
        for enemy in self.enemies:
            if not enemy["alive"]:
                continue

            dist_to_player = np.sum(np.abs(enemy["pos"] - self.robot_pos))

            if dist_to_player <= self.ENEMY_ATTACK_RANGE:
                # Attack
                # // SFX: Robot hit
                self.robot_health -= self.ENEMY_ATTACK_DAMAGE
                damage_taken_reward -= 1.0
                self._create_damage_particle(self.robot_pos, self.ENEMY_ATTACK_DAMAGE, is_player=True)
            else:
                # Move
                # // SFX: Enemy move
                delta = self.robot_pos - enemy["pos"]
                new_pos = enemy["pos"].copy()
                if abs(delta[0]) > abs(delta[1]):
                    new_pos[0] += np.sign(delta[0])
                else:
                    new_pos[1] += np.sign(delta[1])
                
                # Check for collisions with other entities before moving
                is_occupied = False
                if np.array_equal(new_pos, self.robot_pos):
                    is_occupied = True
                for other_enemy in self.enemies:
                    if other_enemy["id"] != enemy["id"] and other_enemy["alive"] and np.array_equal(new_pos, other_enemy["pos"]):
                        is_occupied = True
                        break
                
                if not is_occupied:
                    enemy["pos"] = new_pos
        
        return damage_taken_reward

    def _check_termination(self):
        if self.robot_health <= 0:
            self.game_over = True
            self.win_message = "GAME OVER"
            return True, -100.0
        
        if self._check_win_condition():
            self.game_over = True
            self.win_message = "VICTORY!"
            return True, 100.0

        if self.steps >= self.MAX_STEPS:
            return True, 0.0

        return False, 0.0

    def _check_win_condition(self):
        return all(not enemy["alive"] for enemy in self.enemies)

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
            "robot_health": self.robot_health,
            "enemies_alive": sum(1 for e in self.enemies if e["alive"]),
        }

    def _world_to_screen(self, grid_pos):
        return (
            self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE
        )

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_SIZE * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw enemies
        for enemy in self.enemies:
            if enemy["alive"]:
                self._draw_entity(enemy["pos"], self.COLOR_ENEMY, 0.8)
                self._draw_health_bar(enemy["pos"], enemy["health"], self.ENEMY_MAX_HEALTH)
        
        # Draw robot
        self._draw_entity(self.robot_pos, self.COLOR_ROBOT, 0.9)
        self._draw_health_bar(self.robot_pos, self.robot_health, self.PLAYER_MAX_HEALTH)

        # Draw particles
        for p in self.damage_particles:
            text_surf = self.font_small.render(p["text"], True, p["color"])
            text_rect = text_surf.get_rect(center=p["pos"])
            self.screen.blit(text_surf, text_rect)

    def _draw_entity(self, pos, color, size_ratio=0.8):
        screen_pos = self._world_to_screen(pos)
        entity_size = int(self.CELL_SIZE * size_ratio)
        offset = (self.CELL_SIZE - entity_size) // 2
        rect = pygame.Rect(screen_pos[0] + offset, screen_pos[1] + offset, entity_size, entity_size)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        pygame.draw.rect(self.screen, tuple(min(255, c + 50) for c in color), rect, width=2, border_radius=4)

    def _draw_health_bar(self, pos, current_hp, max_hp):
        screen_pos = self._world_to_screen(pos)
        bar_width = self.CELL_SIZE * 0.8
        bar_height = 5
        x = screen_pos[0] + (self.CELL_SIZE - bar_width) // 2
        y = screen_pos[1] - bar_height - 2
        
        health_ratio = max(0, current_hp / max_hp)
        
        bg_rect = pygame.Rect(x, y, bar_width, bar_height)
        fg_rect = pygame.Rect(x, y, bar_width * health_ratio, bar_height)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect, border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, fg_rect, border_radius=2)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 30))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_surf = self.font_large.render(self.win_message, True, self.COLOR_ROBOT if self._check_win_condition() else self.COLOR_ENEMY)
            end_text_rect = end_text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text_surf, end_text_rect)

    def _create_damage_particle(self, grid_pos, damage, is_player=False):
        screen_pos = self._world_to_screen(grid_pos)
        center_pos = (screen_pos[0] + self.CELL_SIZE // 2, screen_pos[1] + self.CELL_SIZE // 2)
        self.damage_particles.append({
            "text": str(damage),
            "pos": center_pos,
            "color": self.COLOR_ENEMY if is_player else self.COLOR_DAMAGE,
            "lifetime": 30, # frames
            "y_vel": -2
        })

    def _update_particles(self):
        for p in self.damage_particles[:]:
            p["lifetime"] -= 1
            p["pos"] = (p["pos"][0], p["pos"][1] + p["y_vel"])
            alpha = max(0, 255 * (p["lifetime"] / 30.0))
            p["color"] = (p["color"][0], p["color"][1], p["color"][2], alpha)
            if p["lifetime"] <= 0:
                self.damage_particles.remove(p)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        # Reset is needed to initialize state for observation
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Example ---
    # This loop allows a human to play the game.
    pygame.display.set_caption("Robot Combat Grid")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    while not terminated:
        # Convert observation for display
        display_obs = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(display_obs)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Get human input
        action = np.array([0, 0, 0]) # [movement, space, shift]
        
        # This is a blocking call, waiting for an event
        event_processed = False
        while not event_processed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    event_processed = True
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    elif event.key == pygame.K_SPACE:
                        action[1] = 1
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                        action[2] = 1
                    elif event.key == pygame.K_r: # Reset key
                        obs, info = env.reset()
                    
                    # Any key press constitutes a turn
                    event_processed = True
            if terminated:
                break
        
        if terminated:
            break
            
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        if terminated:
            # Display final frame
            display_obs = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(display_obs)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            print("Game Over! Press any key to exit.")
            # Wait for one more key press to close
            waiting_for_exit = True
            while waiting_for_exit:
                 for event in pygame.event.get():
                     if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                         waiting_for_exit = False
                         break

    pygame.quit()