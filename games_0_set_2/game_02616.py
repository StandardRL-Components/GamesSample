
# Generated: 2025-08-27T20:54:35.886748
# Source Brief: brief_02616.md
# Brief Index: 2616

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    A Gymnasium environment for a grid-based dungeon crawler game.
    The player explores a dungeon, collects gold, fights enemies, and tries to
    reach the treasure chest before their health runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Space to attack an adjacent enemy."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore a procedurally generated dungeon, battling enemies and collecting gold to find the treasure."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        except pygame.error:
            self.font = pygame.font.SysFont(None, 30)
            self.font_small = pygame.font.SysFont(None, 24)


        # Game constants
        self.GRID_SIZE = 8
        self.CELL_SIZE = 50
        self.GAME_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GAME_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.X_OFFSET = (640 - self.GAME_WIDTH) // 2
        self.Y_OFFSET = (400 - self.GAME_HEIGHT) // 2
        
        self.MAX_STEPS = 1000
        self.INITIAL_PLAYER_HEALTH = 10
        self.NUM_ENEMIES = 10
        self.NUM_GOLD = 10

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_FLOOR = (87, 56, 40)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_ENEMY = (50, 150, 255)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_CHEST = (218, 165, 32)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_ATTACK = (255, 255, 255)
        self.COLOR_DAMAGE_FLASH = (255, 0, 0, 100)

        # Game state variables (initialized in reset)
        self.player_pos = None
        self.player_health = None
        self.gold_collected = None
        self.enemies = None
        self.gold_piles = None
        self.chest_pos = None
        self.steps = None
        self.terminated = None
        self.last_action_feedback = []
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.gold_collected = 0
        self.terminated = False
        self.player_health = self.INITIAL_PLAYER_HEALTH
        self.last_action_feedback = []

        self._generate_level()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Procedurally places player, enemies, gold, and chest on the grid."""
        all_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(all_coords)

        self.player_pos = all_coords.pop()
        self.chest_pos = all_coords.pop()
        self.gold_piles = [all_coords.pop() for _ in range(self.NUM_GOLD)]
        self.enemies = [{'pos': all_coords.pop(), 'health': 1} for _ in range(self.NUM_ENEMIES)]
    
    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        shift_pressed = action[2] == 1  # Boolean (unused)
        
        self.steps += 1
        self.last_action_feedback.clear()
        
        reward = -0.02  # Base cost for taking a turn/safe move

        # 1. Player Action Phase
        # Player movement
        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
                self.player_pos = new_pos

        # Player attack
        if space_pressed:
            for enemy in self.enemies[:]:
                px, py = self.player_pos
                ex, ey = enemy['pos']
                if abs(px - ex) + abs(py - ey) == 1:  # Adjacent
                    # sound: sword_hit.wav
                    self.last_action_feedback.append(('attack', self.player_pos, enemy['pos']))
                    enemy['health'] -= 1
                    if enemy['health'] <= 0:
                        self.enemies.remove(enemy)
                        reward += 1.0  # Defeated enemy reward
                    break # Only attack one enemy

        # 2. Interaction and Consequence Phase
        # Collect gold
        if self.player_pos in self.gold_piles:
            # sound: coin_pickup.wav
            self.last_action_feedback.append(('collect_gold', self.player_pos, 10))
            self.gold_piles.remove(self.player_pos)
            self.gold_collected += 10
            reward += 0.1

        # Check for victory
        if self.player_pos == self.chest_pos:
            # sound: victory_fanfare.wav
            reward = 100.0  # Goal-oriented reward
            self.terminated = True

        # 3. Enemy Action Phase
        if not self.terminated:
            for enemy in self.enemies:
                px, py = self.player_pos
                ex, ey = enemy['pos']
                if abs(px - ex) + abs(py - ey) == 1:
                    # sound: player_hurt.wav
                    self.last_action_feedback.append(('damage',))
                    self.player_health -= 1

        # 4. Termination Check
        if self.player_health <= 0:
            # sound: game_over.wav
            reward = -100.0  # Failure penalty
            self.terminated = True
        
        if self.steps >= self.MAX_STEPS:
            self.terminated = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.gold_collected,
            "steps": self.steps,
            "health": self.player_health,
        }

    def _render_game(self):
        """Renders the main game grid and its contents."""
        game_surface = pygame.Surface((self.GAME_WIDTH, self.GAME_HEIGHT))
        game_surface.fill(self.COLOR_FLOOR)

        # Draw gold
        for gx, gy in self.gold_piles:
            pygame.draw.circle(
                game_surface, self.COLOR_GOLD, 
                (int(gx * self.CELL_SIZE + self.CELL_SIZE / 2), int(gy * self.CELL_SIZE + self.CELL_SIZE / 2)),
                int(self.CELL_SIZE / 4)
            )
        
        # Draw chest
        cx, cy = self.chest_pos
        pygame.draw.rect(game_surface, self.COLOR_CHEST, 
            (cx * self.CELL_SIZE + 5, cy * self.CELL_SIZE + 15, self.CELL_SIZE - 10, self.CELL_SIZE - 20))
        pygame.draw.rect(game_surface, pygame.Color(self.COLOR_CHEST).lerp((0,0,0), 0.2), 
            (cx * self.CELL_SIZE + 2, cy * self.CELL_SIZE + 10, self.CELL_SIZE - 4, 10))

        # Draw enemies
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            pygame.draw.rect(game_surface, self.COLOR_ENEMY, 
                (ex * self.CELL_SIZE + 8, ey * self.CELL_SIZE + 8, self.CELL_SIZE - 16, self.CELL_SIZE - 16))

        # Draw player
        px, py = self.player_pos
        player_rect = (px * self.CELL_SIZE + 5, py * self.CELL_SIZE + 5, self.CELL_SIZE - 10, self.CELL_SIZE - 10)
        pygame.draw.rect(game_surface, self.COLOR_PLAYER, player_rect, border_radius=3)
        # Player glow effect
        glow_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, self.CELL_SIZE // 2, self.CELL_SIZE // 2, 
                                     self.CELL_SIZE // 2 - 2, (255, 100, 100, 40))
        game_surface.blit(glow_surf, (px * self.CELL_SIZE, py * self.CELL_SIZE))

        # Blit game surface to the main screen
        self.screen.blit(game_surface, (self.X_OFFSET, self.Y_OFFSET))
        
        self._render_feedback()

    def _render_feedback(self):
        """Renders transient visual effects for player actions."""
        for feedback in self.last_action_feedback:
            feedback_type = feedback[0]
            if feedback_type == 'attack':
                _, start_pos, end_pos = feedback
                start_px = (self.X_OFFSET + start_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2, 
                            self.Y_OFFSET + start_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2)
                end_px = (self.X_OFFSET + end_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2, 
                          self.Y_OFFSET + end_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2)
                pygame.draw.line(self.screen, self.COLOR_ATTACK, start_px, end_px, 3)

            elif feedback_type == 'damage':
                flash_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                flash_surface.fill(self.COLOR_DAMAGE_FLASH)
                self.screen.blit(flash_surface, (0, 0))

            elif feedback_type == 'collect_gold':
                _, pos, amount = feedback
                text_surf = self.font_small.render(f"+{amount}", True, self.COLOR_GOLD)
                text_rect = text_surf.get_rect(center=(
                    self.X_OFFSET + pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2,
                    self.Y_OFFSET + pos[1] * self.CELL_SIZE + self.CELL_SIZE // 4
                ))
                self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        """Renders the UI elements like health, gold, and steps."""
        # Health UI
        health_text = self.font.render(f"Health: {max(0, self.player_health)}/{self.INITIAL_PLAYER_HEALTH}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 10))

        # Gold UI
        gold_text = self.font.render(f"Gold: {self.gold_collected}", True, self.COLOR_UI_TEXT)
        gold_rect = gold_text.get_rect(topright=(640 - 15, 10))
        self.screen.blit(gold_text, gold_rect)
        
        # Steps UI
        steps_text = self.font.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        steps_rect = steps_text.get_rect(midtop=(640 / 2, 10))
        self.screen.blit(steps_text, steps_rect)
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("Dungeon Crawler")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    terminated = False
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("Press Q to quit.")
    print("="*30 + "\n")

    while not terminated:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True
                # Movement
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                # Actions
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
        
        # Only step if an action was taken
        if any(action):
            obs, reward, terminated, truncated, info = env.step(np.array(action))
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    env.close()
    print("Game Over!")