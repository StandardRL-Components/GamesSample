
# Generated: 2025-08-27T15:36:02.592070
# Source Brief: brief_01026.md
# Brief Index: 1026

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. Hold shift in designated areas to hide. Press space to search for clues."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore a haunted house, find hidden clues, and escape the killer's grasp in this side-scrolling horror game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1500
        self.NUM_CLUES = 5
        self.PLAYER_SPEED = 3
        self.PLAYER_SIZE = (12, 28)
        self.KILLER_SIZE = (16, 32)
        self.SEARCH_RADIUS = 40
        self.KILLER_DETECT_RADIUS = 50
        self.KILLER_HIDDEN_DETECT_RADIUS = 20
        self.KILLER_CATCH_RADIUS = 10
        self.KILLER_PROXIMITY_SHAKE_THRESHOLD = 150

        # Colors
        self.COLOR_BG = (10, 5, 15)
        self.COLOR_WALL = (30, 25, 40)
        self.COLOR_FLOOR = (20, 15, 30)
        self.COLOR_PLAYER = (220, 220, 255)
        self.COLOR_KILLER = (200, 30, 30)
        self.COLOR_HIDING_SPOT = (40, 50, 60)
        self.COLOR_HIDING_ACTIVE = (50, 180, 50)
        self.COLOR_CLUE_FOUND = (100, 255, 100)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_SEARCH_FX = (255, 255, 255)
        
        # Reward structure
        self.REWARD_STEP_PENALTY = -0.01
        self.REWARD_HIDING = 0.1
        self.REWARD_CLUE = 10
        self.REWARD_CAUGHT = -100
        self.REWARD_WIN = 100

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.killer_pos = np.zeros(2, dtype=np.float32)
        self.killer_speed = 0.0
        self.killer_direction = 1
        self.clue_locations = []
        self.clues_found_mask = []
        self.hiding_spots = []
        self.is_hiding = False
        self.prev_space_held = False
        self.search_effect = {"active": False, "radius": 0, "pos": (0, 0)}
        self.particles = []
        self.screen_shake = 0
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        floor_y = self.HEIGHT - 40
        self.player_pos = np.array([self.WIDTH / 2, floor_y], dtype=np.float32)
        self.killer_pos = np.array([50, floor_y], dtype=np.float32)
        self.killer_speed = 1.0
        self.killer_direction = 1
        
        self.prev_space_held = False
        self.is_hiding = False
        
        # Define static hiding spots and furniture
        self.hiding_spots = [
            pygame.Rect(100, floor_y - 60, 50, 60),
            pygame.Rect(self.WIDTH - 150, floor_y - 60, 50, 60)
        ]
        self.furniture = [
            pygame.Rect(250, floor_y - 40, 140, 40), # Table
            pygame.Rect(20, floor_y - 80, 30, 80), # Tall cabinet
            pygame.Rect(self.WIDTH - 220, floor_y - 50, 30, 50) # Small cabinet
        ]
        
        # Generate clue locations
        possible_clue_zones = [
            (125, floor_y), (180, floor_y - 20), (320, floor_y - 50),
            (450, floor_y), (50, floor_y - 50), (self.WIDTH - 50, floor_y - 20),
            (self.WIDTH - 125, floor_y - 70)
        ]
        
        clue_indices = self.np_random.choice(len(possible_clue_zones), self.NUM_CLUES, replace=False)
        self.clue_locations = [possible_clue_zones[i] for i in clue_indices]
        self.clues_found_mask = [False] * self.NUM_CLUES
        
        # Effects
        self.search_effect["active"] = False
        self.particles = [self._create_particle() for _ in range(50)]
        self.screen_shake = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = self.REWARD_STEP_PENALTY
        
        # --- Player Logic ---
        # Movement
        if movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE[0] / 2, self.WIDTH - self.PLAYER_SIZE[0] / 2)

        # Hiding
        player_rect = self._get_player_rect()
        in_hiding_spot = any(player_rect.colliderect(spot) for spot in self.hiding_spots)
        self.is_hiding = shift_held and in_hiding_spot
        
        # Searching
        search_triggered = space_held and not self.prev_space_held
        if search_triggered:
            # sfx: Search_Sound
            self.search_effect = {"active": True, "radius": 1, "pos": self.player_pos.copy()}
            for i, loc in enumerate(self.clue_locations):
                if not self.clues_found_mask[i]:
                    dist = np.linalg.norm(self.player_pos - loc)
                    if dist < self.SEARCH_RADIUS:
                        self.clues_found_mask[i] = True
                        reward += self.REWARD_CLUE
                        # sfx: Clue_Found_Chime
                        break # Only find one clue per search

        self.prev_space_held = space_held
        
        # --- Killer Logic ---
        self.killer_speed = 1.0 + 0.05 * (self.steps // 500)
        self.killer_pos[0] += self.killer_direction * self.killer_speed
        if self.killer_pos[0] > self.WIDTH - 50 or self.killer_pos[0] < 50:
            self.killer_direction *= -1

        # --- Game State & Rewards ---
        dist_to_killer = np.linalg.norm(self.player_pos - self.killer_pos)
        
        if self.is_hiding and dist_to_killer < self.KILLER_PROXIMITY_SHAKE_THRESHOLD:
            reward += self.REWARD_HIDING

        # Detection
        effective_detect_radius = self.KILLER_HIDDEN_DETECT_RADIUS if self.is_hiding else self.KILLER_DETECT_RADIUS
        if dist_to_killer < effective_detect_radius:
            # Killer is aware, could add visual cue here
            pass

        if dist_to_killer < self.KILLER_CATCH_RADIUS:
            self.game_over = True
            reward = self.REWARD_CAUGHT
            self.win_message = "CAUGHT"
            # sfx: Player_Caught_Scream

        # Win condition
        if all(self.clues_found_mask):
            self.game_over = True
            reward = self.REWARD_WIN
            self.win_message = "ESCAPED!"
            # sfx: Victory_Fanfare

        # Step limit
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            self.win_message = "TIME'S UP"
            terminated = True
        
        self.score += reward
        
        # --- Update Effects ---
        self._update_effects(dist_to_killer)

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_observation(self):
        # Create a temporary surface for screen shake
        render_surface = self.screen.copy()
        render_surface.fill(self.COLOR_BG)
        
        self._render_game(render_surface)
        self._render_ui(render_surface)

        # Apply screen shake
        if self.screen_shake > 0:
            offset_x = random.randint(-int(self.screen_shake), int(self.screen_shake))
            offset_y = random.randint(-int(self.screen_shake), int(self.screen_shake))
            self.screen.blit(render_surface, (offset_x, offset_y))
        else:
            self.screen.blit(render_surface, (0, 0))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self, surface):
        floor_y = self.HEIGHT - 40
        light_level = random.uniform(0.95, 1.0) # Flickering light effect
        
        # Floor and Walls
        pygame.draw.rect(surface, self.COLOR_FLOOR, (0, floor_y, self.WIDTH, 40))
        pygame.draw.rect(surface, self.COLOR_WALL, (0, 0, self.WIDTH, floor_y))
        
        # Hiding spots
        player_rect = self._get_player_rect()
        for spot in self.hiding_spots:
            color = self.COLOR_HIDING_SPOT
            if self.is_hiding and player_rect.colliderect(spot):
                color = self.COLOR_HIDING_ACTIVE
            pygame.draw.rect(surface, color, spot)
            
        # Furniture
        for item in self.furniture:
            pygame.draw.rect(surface, self.COLOR_FLOOR, item)
            pygame.draw.rect(surface, self.COLOR_WALL, item, 2)

        # Particles (dust motes)
        for p in self.particles:
            p_color = (p['color'][0] * light_level, p['color'][1] * light_level, p['color'][2] * light_level)
            pygame.gfxdraw.pixel(surface, int(p['pos'][0]), int(p['pos'][1]), p_color)
            
        # Found Clues
        for i, found in enumerate(self.clues_found_mask):
            if found:
                pos = self.clue_locations[i]
                pygame.draw.circle(surface, self.COLOR_CLUE_FOUND, (int(pos[0]), int(pos[1])), 5)
                # Glow effect
                for j in range(3):
                    alpha = 80 - j * 20
                    pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), 8 + j * 4, (*self.COLOR_CLUE_FOUND, alpha))

        # Killer
        self._render_killer(surface, light_level)

        # Player
        self._render_player(surface, light_level)
        
        # Search Effect
        if self.search_effect["active"]:
            alpha = max(0, 255 - int(self.search_effect["radius"] * (255 / self.SEARCH_RADIUS)))
            pygame.gfxdraw.aacircle(surface, int(self.search_effect["pos"][0]), int(self.search_effect["pos"][1]), int(self.search_effect["radius"]), (*self.COLOR_SEARCH_FX, alpha))

    def _render_player(self, surface, light_level):
        color = tuple(c * light_level for c in self.COLOR_PLAYER)
        rect = self._get_player_rect()
        pygame.draw.rect(surface, color, rect, border_radius=3)
        
    def _render_killer(self, surface, light_level):
        color = tuple(c * light_level for c in self.COLOR_KILLER)
        rect = pygame.Rect(0, 0, self.KILLER_SIZE[0], self.KILLER_SIZE[1])
        rect.midbottom = (int(self.killer_pos[0]), int(self.killer_pos[1]))
        pygame.draw.rect(surface, color, rect, border_radius=3)
        
        # Danger aura
        dist_to_killer = np.linalg.norm(self.player_pos - self.killer_pos)
        aura_intensity = max(0, 1 - (dist_to_killer / self.KILLER_PROXIMITY_SHAKE_THRESHOLD))
        if aura_intensity > 0:
            for i in range(3):
                radius = int(self.KILLER_CATCH_RADIUS + i * 15 * aura_intensity)
                alpha = int(40 * aura_intensity * (1 - i/3))
                if alpha > 0:
                    pygame.gfxdraw.filled_circle(surface, rect.centerx, rect.centery, radius, (*self.COLOR_KILLER, alpha))
    
    def _render_ui(self, surface):
        # Clue count
        clues_found = sum(self.clues_found_mask)
        clue_text = f"Clues: {clues_found} / {self.NUM_CLUES}"
        text_surface = self.font_small.render(clue_text, True, self.COLOR_TEXT)
        surface.blit(text_surface, (10, 10))

        # Game over text
        if self.game_over:
            # Dark overlay
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            surface.blit(overlay, (0, 0))
            
            end_text_surface = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            text_rect = end_text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            surface.blit(end_text_surface, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "clues_found": sum(self.clues_found_mask),
            "distance_to_killer": np.linalg.norm(self.player_pos - self.killer_pos),
        }

    def _get_player_rect(self):
        rect = pygame.Rect(0, 0, self.PLAYER_SIZE[0], self.PLAYER_SIZE[1])
        rect.midbottom = (int(self.player_pos[0]), int(self.player_pos[1]))
        return rect
        
    def _create_particle(self):
        return {
            "pos": [random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)],
            "vel": [random.uniform(-0.1, 0.1), random.uniform(-0.2, -0.05)],
            "lifetime": random.uniform(100, 300),
            "color": (random.randint(40, 60), random.randint(35, 55), random.randint(50, 70))
        }

    def _update_effects(self, dist_to_killer):
        # Update particles
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifetime"] -= 1
            if p["lifetime"] <= 0 or p["pos"][1] < 0:
                self.particles.remove(p)
                self.particles.append(self._create_particle())

        # Update search effect
        if self.search_effect["active"]:
            self.search_effect["radius"] += 4
            if self.search_effect["radius"] > self.SEARCH_RADIUS:
                self.search_effect["active"] = False

        # Update screen shake
        if dist_to_killer < self.KILLER_PROXIMITY_SHAKE_THRESHOLD:
            self.screen_shake = max(self.screen_shake, (1 - dist_to_killer / self.KILLER_PROXIMITY_SHAKE_THRESHOLD) * 5)
        self.screen_shake = max(0, self.screen_shake - 0.2)
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Haunted Escape")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    print(env.user_guide)
    
    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1 # Not used, but for completeness
        if keys[pygame.K_DOWN]: movement = 2 # Not used
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS
        
    print(f"Game Over. Final Score: {total_reward:.2f}, Steps: {info['steps']}")
    
    env.close()