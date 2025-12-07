import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:26:20.923442
# Source Brief: brief_00469.md
# Brief Index: 469
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player teleports between floating platforms
    to collect crystals and evade a pursuing shadow.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Teleport between floating platforms to collect crystals while evading a relentless pursuing shadow. "
        "Manage your energy and survive as long as you can."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to teleport to the nearest platform in that direction."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000

    # --- Colors ---
    COLOR_BG = (10, 10, 20)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (50, 255, 50, 40)
    COLOR_PLATFORM = (150, 150, 160)
    COLOR_CRYSTAL = (0, 180, 255)
    COLOR_CRYSTAL_GLOW = (0, 180, 255, 60)
    COLOR_SHADOW = (70, 0, 120)
    COLOR_SHADOW_TINT = (70, 0, 120, 30)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_ENERGY_BAR = (0, 220, 220)
    COLOR_ENERGY_BAR_BG = (40, 40, 60)

    # --- Game Parameters ---
    PLAYER_RADIUS = 10
    INITIAL_SHADOW_SPEED = 0.5
    SHADOW_SPEED_INCREASE = 0.05
    SHADOW_SPEED_INTERVAL = 50
    ENERGY_PER_CRYSTAL = 25
    ENERGY_DECAY_PER_STEP = 0.05
    MAX_ENERGY = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.platforms_traversed = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.current_platform_idx = 0
        self.platforms = []
        self.crystals = []
        self.shadow_pos = pygame.Vector2(0, 0)
        self.shadow_speed = 0
        self.shadow_radius = 0
        self.player_energy = 0
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.platforms_traversed = 0
        self.game_over = False
        
        self._generate_platforms()
        self.current_platform_idx = 0
        self.player_pos = self.platforms[self.current_platform_idx]['pos'].copy()

        self._place_initial_shadow()
        self.shadow_speed = self.INITIAL_SHADOW_SPEED
        self.shadow_radius = 50
        
        self.player_energy = self.MAX_ENERGY
        
        self.crystals = []
        self._generate_crystals()

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # --- Action Handling ---
        teleported = self._handle_teleport(movement)
        if teleported:
            reward += 1.0  # Reward for successful teleport
            self.platforms_traversed += 1
            if self.platforms_traversed > 0 and self.platforms_traversed % self.SHADOW_SPEED_INTERVAL == 0:
                self.shadow_speed += self.SHADOW_SPEED_INCREASE
                reward += 10.0 # Bonus for milestone

        # --- Game State Updates ---
        self._update_shadows()
        self._update_energy()
        
        crystal_collected = self._update_crystals()
        if crystal_collected:
            reward += 0.1 # Small reward for collecting crystal
            # Sound: Crystal collect sound
        else:
            reward -= 0.01 # Small penalty for not collecting

        self._update_particles()
        
        self.steps += 1
        
        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = False # This environment does not truncate based on time limit in the same way as some others
        if terminated and not self.game_over:
            self.game_over = True
            # Sound: Game over sound
            if self.steps >= self.MAX_STEPS:
                reward = 100.0 # Win condition
            else:
                reward = -100.0 # Loss condition
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "platforms_traversed": self.platforms_traversed,
            "steps": self.steps,
            "energy": self.player_energy,
            "shadow_speed": self.shadow_speed,
        }
        
    def _place_initial_shadow(self):
        # Place shadow far from the player's starting corner
        start_platform_pos = self.platforms[0]['pos']
        if start_platform_pos.x < self.SCREEN_WIDTH / 2 and start_platform_pos.y < self.SCREEN_HEIGHT / 2:
            self.shadow_pos = pygame.Vector2(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        elif start_platform_pos.x > self.SCREEN_WIDTH / 2 and start_platform_pos.y < self.SCREEN_HEIGHT / 2:
            self.shadow_pos = pygame.Vector2(0, self.SCREEN_HEIGHT)
        elif start_platform_pos.x < self.SCREEN_WIDTH / 2 and start_platform_pos.y > self.SCREEN_HEIGHT / 2:
            self.shadow_pos = pygame.Vector2(self.SCREEN_WIDTH, 0)
        else:
            self.shadow_pos = pygame.Vector2(0, 0)

    def _generate_platforms(self):
        self.platforms = []
        complexity_factor = 1 + (self.platforms_traversed // 200) * 0.1
        num_platforms = int(10 * complexity_factor)
        min_size = max(20, 40 / complexity_factor)
        max_size = max(40, 60 / complexity_factor)
        
        # Create starting platform
        start_pos = pygame.Vector2(
            self.np_random.integers(50, self.SCREEN_WIDTH - 50),
            self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
        )
        self.platforms.append(self._create_platform(start_pos, self.np_random.uniform(min_size, max_size) * 1.5))
        
        # Create other platforms
        for _ in range(num_platforms - 1):
            placed = False
            for _ in range(20): # Try 20 times to place a platform
                ref_platform = self.np_random.choice(self.platforms)
                angle = self.np_random.uniform(0, 2 * math.pi)
                dist = self.np_random.uniform(100, 250)
                new_pos = ref_platform['pos'] + pygame.Vector2(math.cos(angle), math.sin(angle)) * dist
                
                # Check bounds
                if not (50 < new_pos.x < self.SCREEN_WIDTH - 50 and 50 < new_pos.y < self.SCREEN_HEIGHT - 50):
                    continue

                # Check overlap
                is_overlapping = any(
                    (p['pos'] - new_pos).length() < p['radius'] + max_size for p in self.platforms
                )
                if not is_overlapping:
                    self.platforms.append(self._create_platform(new_pos, self.np_random.uniform(min_size, max_size)))
                    placed = True
                    break
            if not placed:
                break # Stop if we can't place more platforms

    def _create_platform(self, pos, radius):
        num_vertices = self.np_random.integers(5, 9)
        vertices = []
        for i in range(num_vertices):
            angle = (i / num_vertices) * 2 * math.pi
            rad = radius * self.np_random.uniform(0.8, 1.2)
            vertices.append(pygame.Vector2(math.cos(angle), math.sin(angle)) * rad)
            
        return {
            'pos': pos,
            'radius': radius,
            'vertices': vertices,
            'rotation': self.np_random.uniform(0, 2 * math.pi),
            'rotation_speed': self.np_random.uniform(-0.005, 0.005)
        }

    def _generate_crystals(self):
        self.crystals = []
        crystal_platforms = self.np_random.choice(
            range(len(self.platforms)),
            size=min(len(self.platforms), 3 + self.platforms_traversed // 100),
            replace=False
        )
        
        for i in crystal_platforms:
            if i == self.current_platform_idx: continue
            self.crystals.append({
                'platform_idx': i,
                'pos': self.platforms[i]['pos'].copy(),
                'pulse_offset': self.np_random.uniform(0, math.pi)
            })
        
        # Ensure at least one crystal is reachable
        if not any(c for c in self.crystals if self._is_platform_reachable(c['platform_idx'])):
             reachable_platforms = [i for i, _ in enumerate(self.platforms) if i != self.current_platform_idx and self._is_platform_reachable(i)]
             if reachable_platforms:
                 idx = self.np_random.choice(reachable_platforms)
                 self.crystals.append({
                     'platform_idx': idx,
                     'pos': self.platforms[idx]['pos'].copy(),
                     'pulse_offset': self.np_random.uniform(0, math.pi)
                 })


    def _is_platform_reachable(self, target_idx):
        for move in range(1, 5):
            if self._find_teleport_target(move) == target_idx:
                return True
        return False

    def _handle_teleport(self, movement):
        if movement == 0:
            return False

        target_idx = self._find_teleport_target(movement)
        
        if target_idx is not None and target_idx != self.current_platform_idx:
            old_pos = self.player_pos.copy()
            self.current_platform_idx = target_idx
            self.player_pos = self.platforms[target_idx]['pos'].copy()
            
            # Sound: Teleport whoosh
            self._create_teleport_particles(old_pos, self.player_pos)
            return True
            
        return False

    def _find_teleport_target(self, movement):
        current_pos = self.platforms[self.current_platform_idx]['pos']
        candidates = []

        for i, p in enumerate(self.platforms):
            if i == self.current_platform_idx:
                continue
            
            direction = (p['pos'] - current_pos)
            if direction.length() == 0: continue
            
            is_candidate = False
            # Normalize direction for angle checks
            dir_norm = direction.normalize()

            if movement == 1 and dir_norm.y < -0.5: # Up
                is_candidate = True
            elif movement == 2 and dir_norm.y > 0.5: # Down
                is_candidate = True
            elif movement == 3 and dir_norm.x < -0.5: # Left
                is_candidate = True
            elif movement == 4 and dir_norm.x > 0.5: # Right
                is_candidate = True
            
            if is_candidate:
                candidates.append((direction.length_squared(), i))
        
        if not candidates:
            return None
        
        # Find the closest candidate(s)
        min_dist_sq = min(c[0] for c in candidates)
        closest_candidates = [c[1] for c in candidates if c[0] == min_dist_sq]
        
        # If multiple are equidistant, choose one randomly
        return self.np_random.choice(closest_candidates)


    def _update_shadows(self):
        if self.player_pos != self.shadow_pos:
            self.shadow_pos.move_towards_ip(self.player_pos, self.shadow_speed)

    def _update_energy(self):
        self.player_energy = max(0, self.player_energy - self.ENERGY_DECAY_PER_STEP)

    def _update_crystals(self):
        collected = False
        crystal_to_remove = None
        for i, crystal in enumerate(self.crystals):
            if crystal['platform_idx'] == self.current_platform_idx:
                self.player_energy = min(self.MAX_ENERGY, self.player_energy + self.ENERGY_PER_CRYSTAL)
                crystal_to_remove = i
                collected = True
                break
        
        if crystal_to_remove is not None:
            self.crystals.pop(crystal_to_remove)
            if not self.crystals:
                self._generate_crystals() # Replenish crystals
        
        return collected

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['vel'] *= 0.95 # Damping

    def _check_termination(self):
        # 1. Caught by shadow
        if (self.player_pos - self.shadow_pos).length() < self.PLAYER_RADIUS + self.shadow_radius * 0.5:
            return True
        # 2. Energy depleted
        if self.player_energy <= 0:
            return True
        # 3. Max steps reached
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _create_teleport_particles(self, start_pos, end_pos):
        # Burst from start
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            self.particles.append({
                'pos': start_pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'lifespan': self.np_random.integers(15, 30),
                'color': self.COLOR_PLAYER,
                'size': self.np_random.uniform(1, 4)
            })
        # Implode at end
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            self.particles.append({
                'pos': end_pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * 50,
                'vel': (end_pos - (end_pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * 50)) / 15,
                'lifespan': self.np_random.integers(10, 20),
                'color': self.COLOR_PLAYER,
                'size': self.np_random.uniform(1, 4)
            })

    def _render_game(self):
        self._render_shadow()
        self._render_platforms()
        self._render_crystals()
        self._render_particles()
        self._render_player()

    def _render_shadow(self):
        # Pulsating effect for the shadow
        pulse = math.sin(self.steps * 0.05) * 5
        base_radius = self.shadow_radius + pulse
        pos = (int(self.shadow_pos.x), int(self.shadow_pos.y))
        
        # Draw multiple semi-transparent circles for an amorphous look
        for i in range(5, 0, -1):
            radius = int(base_radius * (1 + (i-3)*0.1) + math.sin(self.steps * 0.05 + i) * 3)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_SHADOW_TINT)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(base_radius * 0.7), self.COLOR_SHADOW)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(base_radius * 0.7), self.COLOR_SHADOW)


    def _render_platforms(self):
        for p in self.platforms:
            p['rotation'] += p['rotation_speed']
            rotated_vertices = [v.rotate_rad(p['rotation']) + p['pos'] for v in p['vertices']]
            int_vertices = [(int(v.x), int(v.y)) for v in rotated_vertices]
            pygame.gfxdraw.aapolygon(self.screen, int_vertices, self.COLOR_PLATFORM)
            pygame.gfxdraw.filled_polygon(self.screen, int_vertices, self.COLOR_PLATFORM)

    def _render_crystals(self):
        for c in self.crystals:
            pulse = (math.sin(self.steps * 0.1 + c['pulse_offset']) + 1) / 2
            size = int(6 + pulse * 4)
            pos = (int(c['pos'].x), int(c['pos'].y))
            
            # Glow
            glow_radius = int(size * 2.5)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, self.COLOR_CRYSTAL_GLOW)
            
            # Core
            rect = pygame.Rect(pos[0] - size//2, pos[1] - size//2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_CRYSTAL, rect)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color = (*p['color'], max(0, min(255, alpha)))
            pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), color)

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Glow effect
        glow_radius = int(self.PLAYER_RADIUS * 2.5)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Player triangle
        points = [
            (pos[0], pos[1] - self.PLAYER_RADIUS),
            (pos[0] - self.PLAYER_RADIUS / 1.15, pos[1] + self.PLAYER_RADIUS / 2),
            (pos[0] + self.PLAYER_RADIUS / 1.15, pos[1] + self.PLAYER_RADIUS / 2)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Energy Bar
        bar_width = 150
        bar_height = 15
        energy_ratio = self.player_energy / self.MAX_ENERGY
        current_width = int(bar_width * energy_ratio)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR_BG, (10, 10, bar_width, bar_height))
        if current_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR, (10, 10, current_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, 10, bar_width, bar_height), 1)

        # Traversed Platforms (Score)
        score_text = self.font_ui.render(f"SCORE: {self.platforms_traversed}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "ESCAPE COMPLETE" if self.steps >= self.MAX_STEPS else "OVERTAKEN"
            end_text = self.font_game_over.render(end_text_str, True, self.COLOR_UI_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, end_rect)


    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-comment the following line to run with a display
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Shadow Jumper")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset")
    print("Q: Quit")

    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Env Reset ---")
                
                # Map keys to actions
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['platforms_traversed']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()