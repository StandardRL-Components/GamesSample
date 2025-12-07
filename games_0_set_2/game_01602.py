
# Generated: 2025-08-27T17:38:53.346868
# Source Brief: brief_01602.md
# Brief Index: 1602

        
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
    A Gymnasium environment for a fast-paced arcade game. The player must collect
    sparkling gems while avoiding patrolling enemies, all under a time limit.
    The game prioritizes visual quality and engaging gameplay with a risk/reward
    scoring system.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. "
        "Collect all the gems while avoiding the red enemies."
    )

    # User-facing game description
    game_description = (
        "Collect sparkling gems while dodging cunning enemies in a "
        "fast-paced, top-down arcade game."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 150, 50)
    COLOR_GEM = (0, 191, 255)
    COLOR_GEM_GLOW = (100, 220, 255, 80)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_UI = (255, 255, 255)
    COLOR_WALL = (200, 200, 220)
    WALL_THICKNESS = 10

    # Entity properties
    PLAYER_SIZE = 20
    PLAYER_SPEED = 6
    GEM_SIZE = 8
    NUM_GEMS = 20
    ENEMY_SIZE = 24
    ENEMY_SPEED = 2.5
    NUM_ENEMIES = 3
    DANGER_RADIUS = 80 # For risky gem collection bonus

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)

        # Game state variables (initialized in reset)
        self.player_rect = None
        self.enemies = []
        self.gems = []
        self.steps = 0
        self.score = 0
        self.collected_gems_count = 0
        self.timer = 0
        self.game_over = False
        self.game_outcome = ""

        # Playable area
        self.bounds_rect = pygame.Rect(
            self.WALL_THICKNESS,
            self.WALL_THICKNESS,
            self.SCREEN_WIDTH - 2 * self.WALL_THICKNESS,
            self.SCREEN_HEIGHT - 2 * self.WALL_THICKNESS
        )
        
        # Validate implementation after setup
        # self.validate_implementation() # Uncomment for testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.collected_gems_count = 0
        self.timer = self.MAX_STEPS
        self.game_over = False
        self.game_outcome = ""

        # Player initialization
        self.player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_rect.center = self.bounds_rect.center

        # Enemy initialization
        self.enemies = []
        for _ in range(self.NUM_ENEMIES):
            self._spawn_enemy()

        # Gem initialization
        self.gems = []
        for i in range(self.NUM_GEMS):
            self._spawn_gem(i)

        return self._get_observation(), self._get_info()

    def _spawn_enemy(self):
        patrol_w = self.np_random.integers(150, 300)
        patrol_h = self.np_random.integers(100, 200)
        patrol_x = self.np_random.integers(self.bounds_rect.left, self.bounds_rect.right - patrol_w)
        patrol_y = self.np_random.integers(self.bounds_rect.top, self.bounds_rect.bottom - patrol_h)
        
        path_rect = pygame.Rect(patrol_x, patrol_y, patrol_w, patrol_h)
        corners = [path_rect.topleft, path_rect.topright, path_rect.bottomright, path_rect.bottomleft]
        start_corner_idx = self.np_random.integers(0, 4)
        
        enemy_rect = pygame.Rect(0, 0, self.ENEMY_SIZE, self.ENEMY_SIZE)
        enemy_rect.center = corners[start_corner_idx]
        
        self.enemies.append({
            "rect": enemy_rect,
            "path_corners": corners,
            "target_idx": (start_corner_idx + 1) % 4
        })

    def _spawn_gem(self, index):
        while True:
            pos_x = self.np_random.integers(self.bounds_rect.left, self.bounds_rect.right - self.GEM_SIZE)
            pos_y = self.np_random.integers(self.bounds_rect.top, self.bounds_rect.bottom - self.GEM_SIZE)
            gem_rect = pygame.Rect(pos_x, pos_y, self.GEM_SIZE, self.GEM_SIZE)
            
            # Ensure gems don't overlap with each other or the player start
            if not gem_rect.colliderect(self.player_rect) and not any(gem_rect.colliderect(g['rect']) for g in self.gems):
                self.gems.append({
                    "rect": gem_rect,
                    "sparkle_phase": self.np_random.random() * 2 * math.pi
                })
                break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Game Logic ---
        reward = self._update_game_state(action)
        self.steps += 1
        self.timer -= 1
        
        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            reward += self._get_terminal_reward()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self, action):
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        # Store pre-move distances for reward calculation
        prev_dist_gem, _ = self._get_closest_entity(self.gems)
        prev_dist_enemy, _ = self._get_closest_entity(self.enemies)
        
        # Move player
        self._move_player(movement)
        
        # Move enemies
        self._move_enemies()
        
        # --- Interactions and Rewards ---
        reward = 0
        
        # Gem collection
        collected_this_step = []
        for gem in self.gems:
            if self.player_rect.colliderect(gem['rect']):
                collected_this_step.append(gem)
                self.collected_gems_count += 1
                # SFX: Gem collect sound
                
                # Base reward for collecting a gem
                reward += 10
                
                # Bonus for risky collection
                _, closest_enemy = self._get_closest_entity(self.enemies, self.player_rect.center)
                if closest_enemy and self._get_distance(self.player_rect.center, closest_enemy['rect'].center) < self.DANGER_RADIUS:
                    reward += 20 # Risky bonus
        
        if collected_this_step:
            self.gems = [g for g in self.gems if g not in collected_this_step]
            self.score += reward

        # Continuous rewards
        new_dist_gem, _ = self._get_closest_entity(self.gems)
        new_dist_enemy, _ = self._get_closest_entity(self.enemies)

        if new_dist_gem is not None and prev_dist_gem is not None:
            if new_dist_gem < prev_dist_gem:
                reward += 1.0 # Moved closer to a gem
        
        if new_dist_enemy is not None and prev_dist_enemy is not None:
            if new_dist_enemy < prev_dist_enemy:
                reward -= 0.1 # Moved closer to an enemy
            elif new_dist_enemy > prev_dist_enemy:
                reward -= 5.0 # Moved away from an enemy (safe play penalty)
        
        return reward

    def _move_player(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -self.PLAYER_SPEED # Up
        elif movement == 2: dy = self.PLAYER_SPEED # Down
        elif movement == 3: dx = -self.PLAYER_SPEED # Left
        elif movement == 4: dx = self.PLAYER_SPEED # Right
        
        self.player_rect.move_ip(dx, dy)
        self.player_rect.clamp_ip(self.bounds_rect)

    def _move_enemies(self):
        for enemy in self.enemies:
            target_pos = enemy["path_corners"][enemy["target_idx"]]
            direction = pygame.math.Vector2(target_pos) - pygame.math.Vector2(enemy["rect"].center)
            
            if direction.length() < self.ENEMY_SPEED:
                enemy["rect"].center = target_pos
                enemy["target_idx"] = (enemy["target_idx"] + 1) % 4
            else:
                direction.normalize_ip()
                enemy["rect"].move_ip(direction * self.ENEMY_SPEED)

    def _check_termination(self):
        # Player-Enemy collision
        for enemy in self.enemies:
            if self.player_rect.colliderect(enemy['rect']):
                self.game_outcome = "CAUGHT!"
                # SFX: Player death sound
                return True
        
        # Win condition
        if self.collected_gems_count >= self.NUM_GEMS:
            self.game_outcome = "YOU WIN!"
            # SFX: Win fanfare
            return True
            
        # Timeout
        if self.timer <= 0:
            self.game_outcome = "TIME UP"
            # SFX: Timeout buzzer
            return True
            
        # Max steps (failsafe)
        if self.steps >= self.MAX_STEPS:
            return True

        return False
    
    def _get_terminal_reward(self):
        if self.game_outcome == "YOU WIN!":
            return 100
        elif self.game_outcome == "CAUGHT!":
            return -100
        elif self.game_outcome == "TIME UP":
            return -50
        return 0

    def _get_distance(self, pos1, pos2):
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

    def _get_closest_entity(self, entity_list, target_pos=None):
        if not entity_list:
            return None, None
        
        if target_pos is None:
            target_pos = self.player_rect.center
        
        closest_dist = float('inf')
        closest_entity = None
        
        for entity in entity_list:
            dist = self._get_distance(target_pos, entity['rect'].center)
            if dist < closest_dist:
                closest_dist = dist
                closest_entity = entity
                
        return closest_dist, closest_entity

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, self.bounds_rect, self.WALL_THICKNESS)
        
        # Gems
        for gem in self.gems:
            center_x, center_y = gem['rect'].center
            # Pulsing glow effect
            glow_radius = int(self.GEM_SIZE * (1.5 + 0.5 * math.sin(self.steps * 0.2 + gem['sparkle_phase'])))
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, self.COLOR_GEM_GLOW)
            # Solid gem
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.GEM_SIZE, self.COLOR_GEM)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.GEM_SIZE, self.COLOR_GEM)
            
        # Enemies
        for enemy in self.enemies:
            # Draw anti-aliased triangles
            p1 = (enemy['rect'].midtop)
            p2 = (enemy['rect'].bottomleft)
            p3 = (enemy['rect'].bottomright)
            points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

        # Player glow
        glow_surf = pygame.Surface((self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_SIZE, self.PLAYER_SIZE), self.PLAYER_SIZE)
        self.screen.blit(glow_surf, (self.player_rect.x - self.PLAYER_SIZE / 2, self.player_rect.y - self.PLAYER_SIZE / 2))
        
        # Player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (20, 20))
        
        # Timer
        time_left = max(0, self.timer // self.FPS)
        timer_text = self.font_ui.render(f"Time: {time_left}", True, self.COLOR_UI)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(timer_text, timer_rect)

        # Gem Counter
        gem_text = self.font_ui.render(f"Gems: {self.collected_gems_count}/{self.NUM_GEMS}", True, self.COLOR_UI)
        gem_rect = gem_text.get_rect(midtop=(self.SCREEN_WIDTH // 2, 20))
        self.screen.blit(gem_text, gem_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_game_over.render(self.game_outcome, True, self.COLOR_UI)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "gems_collected": self.collected_gems_count,
            "outcome": self.game_outcome if self.game_over else "in_progress"
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify the implementation."""
        print("Running implementation validation...")
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


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("-" * 20)
                print("Resetting environment...")
                total_reward = 0
                env.reset()

        if terminated or truncated:
            print(f"Episode finished. Final Info: {info}")
            print(f"Total Reward: {total_reward}")
            # Wait for 'R' to be pressed to restart
            pass
            
        clock.tick(GameEnv.FPS)
        
    env.close()