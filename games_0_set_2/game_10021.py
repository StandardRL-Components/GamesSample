import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:41:27.200009
# Source Brief: brief_00021.md
# Brief Index: 21
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Infiltrate a secure facility as a magnetic agent. Use repulsion to move, collect special parts to "
        "unlock the exit, and activate your cloak to evade guards."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to apply a repulsion force for movement. "
        "Hold space to activate your cloaking device."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_WALL = (40, 50, 60)
    COLOR_DOOR = (60, 70, 80)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_CLOAKED = (100, 200, 255)
    COLOR_GUARD = (255, 50, 50)
    COLOR_GUARD_VISION = (150, 50, 50, 50) # RGBA
    COLOR_PART_REGULAR = (50, 255, 50)
    COLOR_PART_SPECIAL = (255, 255, 0)
    COLOR_EXIT = (255, 200, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BAR_BG = (50, 50, 50)
    
    # Player
    PLAYER_RADIUS = 10
    PLAYER_REPULSION_FORCE = 2.5
    PLAYER_FRICTION = 0.95
    MAX_CLOAK_ENERGY = 100.0
    CLOAK_DRAIN_RATE = 1.0
    CLOAK_RECHARGE_RATE = 0.4

    # Guards
    GUARD_RADIUS = 12
    GUARD_SPEED = 1.5
    GUARD_DETECTION_RADIUS = 80

    # Parts
    PART_RADIUS = 5

    # Game
    MAX_STEPS = 1500
    TOTAL_SPECIAL_PARTS = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Internal State Initialization ---
        self.player_pos = None
        self.player_vel = None
        self.is_cloaked = None
        self.cloak_energy = None
        self.parts = None
        self.guards = None
        self.walls = None
        self.doors = None
        self.exit_zone = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.special_parts_collected = None
        self.particles = []
        self.last_dist_to_exit = float('inf')
        
        # Initialize state variables for the first time
        # self.reset() is called by the environment wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.special_parts_collected = 0
        self.particles.clear()

        # Player
        self.player_pos = np.array([50.0, self.SCREEN_HEIGHT / 2.0])
        self.player_vel = np.array([0.0, 0.0])
        self.is_cloaked = False
        self.cloak_energy = self.MAX_CLOAK_ENERGY

        # --- Level Generation ---
        self._generate_level()

        # Guards
        self._update_guards_and_doors()

        self.last_dist_to_exit = self._get_dist_to_exit(self.player_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Unpack Action ---
        movement, space_held, _ = action
        space_held = space_held == 1

        # --- Update Player State ---
        self._update_cloak(space_held)
        self._apply_movement(movement)
        self._update_player_position()
        
        # --- Update Game World ---
        self._update_guards()
        self._update_particles()
        
        # --- Interactions and Rewards ---
        reward += self._handle_part_collection()
        
        # Distance to exit reward
        current_dist_to_exit = self._get_dist_to_exit(self.player_pos)
        if current_dist_to_exit < self.last_dist_to_exit:
            reward += 1.0
        self.last_dist_to_exit = current_dist_to_exit

        # Survival reward
        reward += 0.1

        # Cloak usage penalty
        if self.is_cloaked and not self._is_guard_nearby():
            reward -= 0.1

        # --- Check Termination Conditions ---
        terminated = False
        detection_result = self._check_guard_detection()
        if detection_result:
            reward = -100.0
            terminated = True
        
        win_result, win_reward = self._check_win_condition()
        if win_result:
            reward += win_reward
            terminated = True

        truncated = self.steps >= self.MAX_STEPS
        
        self.game_over = terminated or truncated
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # =================================================================
    #               STATE UPDATE & LOGIC METHODS
    # =================================================================

    def _update_cloak(self, space_held):
        if space_held and self.cloak_energy > 0:
            self.is_cloaked = True
            self.cloak_energy = max(0, self.cloak_energy - self.CLOAK_DRAIN_RATE)
        else:
            self.is_cloaked = False
            self.cloak_energy = min(self.MAX_CLOAK_ENERGY, self.cloak_energy + self.CLOAK_RECHARGE_RATE)
        
        if self.cloak_energy <= 0:
            self.is_cloaked = False

    def _apply_movement(self, movement):
        repulsion_vec = np.array([0.0, 0.0])
        if movement == 1:  # Up
            repulsion_vec[1] = -self.PLAYER_REPULSION_FORCE
        elif movement == 2:  # Down
            repulsion_vec[1] = self.PLAYER_REPULSION_FORCE
        elif movement == 3:  # Left
            repulsion_vec[0] = -self.PLAYER_REPULSION_FORCE
        elif movement == 4:  # Right
            repulsion_vec[0] = self.PLAYER_REPULSION_FORCE
        
        if movement != 0:
            self.player_vel += repulsion_vec
            # SFX: Magnetic repulsion hum
            self._spawn_particles(self.player_pos, -repulsion_vec)

    def _update_player_position(self):
        # Update position
        self.player_pos += self.player_vel
        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION

        # Wall collisions
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_RADIUS, self.player_pos[1] - self.PLAYER_RADIUS, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
        all_obstacles = self.walls + self.doors
        for wall in all_obstacles:
            if player_rect.colliderect(wall):
                # Horizontal collision
                if player_rect.centerx < wall.centerx and self.player_vel[0] > 0: # Hit from left
                    player_rect.right = wall.left
                    self.player_vel[0] *= -0.5
                elif player_rect.centerx > wall.centerx and self.player_vel[0] < 0: # Hit from right
                    player_rect.left = wall.right
                    self.player_vel[0] *= -0.5
                
                # Vertical collision
                if player_rect.centery < wall.centery and self.player_vel[1] > 0: # Hit from top
                    player_rect.bottom = wall.top
                    self.player_vel[1] *= -0.5
                elif player_rect.centery > wall.centery and self.player_vel[1] < 0: # Hit from bottom
                    player_rect.top = wall.bottom
                    self.player_vel[1] *= -0.5

                self.player_pos = np.array(player_rect.center, dtype=float)

        # Boundary collisions
        if self.player_pos[0] < self.PLAYER_RADIUS:
            self.player_pos[0] = self.PLAYER_RADIUS
            self.player_vel[0] *= -0.5
        if self.player_pos[0] > self.SCREEN_WIDTH - self.PLAYER_RADIUS:
            self.player_pos[0] = self.SCREEN_WIDTH - self.PLAYER_RADIUS
            self.player_vel[0] *= -0.5
        if self.player_pos[1] < self.PLAYER_RADIUS:
            self.player_pos[1] = self.PLAYER_RADIUS
            self.player_vel[1] *= -0.5
        if self.player_pos[1] > self.SCREEN_HEIGHT - self.PLAYER_RADIUS:
            self.player_pos[1] = self.SCREEN_HEIGHT - self.PLAYER_RADIUS
            self.player_vel[1] *= -0.5

    def _update_guards(self):
        for guard in self.guards:
            target_pos = np.array(guard['waypoints'][guard['target_idx']])
            direction = target_pos - guard['pos']
            dist = np.linalg.norm(direction)
            
            if dist < self.GUARD_SPEED:
                guard['target_idx'] = (guard['target_idx'] + 1) % len(guard['waypoints'])
            else:
                guard['pos'] += (direction / dist) * self.GUARD_SPEED

    def _handle_part_collection(self):
        reward = 0
        player_rect = pygame.Rect(0, 0, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
        player_rect.center = tuple(self.player_pos)

        for part in self.parts:
            if not part['collected']:
                part_pos = part['pos']
                dist = np.linalg.norm(self.player_pos - np.array(part_pos))
                if dist < self.PLAYER_RADIUS + self.PART_RADIUS:
                    part['collected'] = True
                    # SFX: Part collection ping
                    if part['type'] == 'special':
                        self.special_parts_collected += 1
                        reward += 5.0
                        self._update_guards_and_doors()
                    else:
                        reward += 0.5
        return reward

    def _check_guard_detection(self):
        if self.is_cloaked:
            return False
        for guard in self.guards:
            dist = np.linalg.norm(self.player_pos - guard['pos'])
            if dist < self.GUARD_DETECTION_RADIUS:
                # Line of sight check
                has_los = True
                all_obstacles = self.walls + self.doors
                for wall in all_obstacles:
                    if wall.clipline(tuple(self.player_pos), tuple(guard['pos'])):
                        has_los = False
                        break
                if has_los:
                    # SFX: Detection alert sound
                    return True
        return False

    def _check_win_condition(self):
        if self.special_parts_collected == self.TOTAL_SPECIAL_PARTS:
            player_rect = pygame.Rect(0, 0, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
            player_rect.center = tuple(self.player_pos)
            if player_rect.colliderect(self.exit_zone):
                # SFX: Level complete fanfare
                return True, 100.0 # Final win reward
            else:
                 # Reaching exit zone is a good intermediate goal
                 if self._get_dist_to_exit(self.player_pos) < 50:
                     return False, 50.0
        return False, 0.0

    # =================================================================
    #                      RENDERING METHODS
    # =================================================================

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Walls and Doors
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
        for door in self.doors:
            pygame.draw.rect(self.screen, self.COLOR_DOOR, door)

        # Exit Zone
        if self.special_parts_collected == self.TOTAL_SPECIAL_PARTS:
            s = pygame.Surface((self.exit_zone.width, self.exit_zone.height), pygame.SRCALPHA)
            alpha = int(100 + 50 * math.sin(self.steps * 0.1))
            s.fill((self.COLOR_EXIT[0], self.COLOR_EXIT[1], self.COLOR_EXIT[2], alpha))
            self.screen.blit(s, self.exit_zone.topleft)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_zone, 2)
        
        # Parts
        for part in self.parts:
            if not part['collected']:
                color = self.COLOR_PART_SPECIAL if part['type'] == 'special' else self.COLOR_PART_REGULAR
                pygame.gfxdraw.aacircle(self.screen, int(part['pos'][0]), int(part['pos'][1]), self.PART_RADIUS, color)
                pygame.gfxdraw.filled_circle(self.screen, int(part['pos'][0]), int(part['pos'][1]), self.PART_RADIUS, color)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*self.COLOR_PLAYER[:3], alpha)
            size = int(self.PLAYER_RADIUS * 0.5 * (p['life'] / p['max_life']))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (p['pos'][0] - size, p['pos'][1] - size))

        # Guards
        for guard in self.guards:
            pos = (int(guard['pos'][0]), int(guard['pos'][1]))
            # Vision cone
            if not self.is_cloaked:
                vision_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(vision_surf, pos[0], pos[1], self.GUARD_DETECTION_RADIUS, self.COLOR_GUARD_VISION)
                self.screen.blit(vision_surf, (0,0))
            # Guard body
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.GUARD_RADIUS, self.COLOR_GUARD)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.GUARD_RADIUS, self.COLOR_GUARD)

        # Player
        player_color = self.COLOR_PLAYER_CLOAKED if self.is_cloaked else self.COLOR_PLAYER
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        # Glow effect
        for i in range(4):
            alpha = 60 - i * 15
            radius = self.PLAYER_RADIUS + i * 3
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*player_color, alpha))
        # Player body
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, player_color)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, player_color)

    def _render_ui(self):
        # Part counter
        part_text = f"SPECIAL PARTS: {self.special_parts_collected} / {self.TOTAL_SPECIAL_PARTS}"
        text_surf = self.font_small.render(part_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Cloak energy bar
        bar_width = 150
        bar_height = 12
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 30, bar_width, bar_height))
        energy_width = int(bar_width * (self.cloak_energy / self.MAX_CLOAK_ENERGY))
        if energy_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_CLOAKED, (10, 30, energy_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, 30, bar_width, bar_height), 1)

        # Game Over Text
        if self.game_over:
            is_win = self.special_parts_collected == self.TOTAL_SPECIAL_PARTS and not self.steps >= self.MAX_STEPS
            msg = "MISSION COMPLETE" if is_win else "DETECTED"
            if self.steps >= self.MAX_STEPS and not is_win:
                msg = "TIME UP"
            color = self.COLOR_PART_SPECIAL if is_win else self.COLOR_GUARD
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    # =================================================================
    #                      HELPER METHODS
    # =================================================================
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "special_parts_collected": self.special_parts_collected,
            "cloak_energy": self.cloak_energy
        }

    def _generate_level(self):
        self.walls = [
            # Boundaries
            pygame.Rect(0, 0, self.SCREEN_WIDTH, 10),
            pygame.Rect(0, self.SCREEN_HEIGHT - 10, self.SCREEN_WIDTH, 10),
            pygame.Rect(0, 0, 10, self.SCREEN_HEIGHT),
            pygame.Rect(self.SCREEN_WIDTH - 10, 0, 10, self.SCREEN_HEIGHT),
            # Internal walls
            pygame.Rect(150, 0, 20, 150),
            pygame.Rect(150, 250, 20, 150),
            pygame.Rect(350, 100, 20, 200),
        ]
        self._all_doors = [
            pygame.Rect(150, 150, 20, 100), # Door 1
            pygame.Rect(350, 0, 20, 100), # Door 2
            pygame.Rect(350, 300, 20, 100), # Door 3 (Not a real door, just a wall)
        ]
        self.parts = [
            {'pos': (250, 50), 'type': 'regular', 'collected': False},
            {'pos': (250, 350), 'type': 'regular', 'collected': False},
            {'pos': (500, 200), 'type': 'regular', 'collected': False},
            {'pos': (100, 100), 'type': 'special', 'collected': False}, # Part 1
            {'pos': (250, 200), 'type': 'special', 'collected': False}, # Part 2
            {'pos': (550, 350), 'type': 'special', 'collected': False}, # Part 3
        ]
        self.exit_zone = pygame.Rect(self.SCREEN_WIDTH - 60, self.SCREEN_HEIGHT/2 - 25, 50, 50)
        self._guard_definitions = [
            {'waypoints': [(250, 100), (250, 300)]},
            {'waypoints': [(450, 50), (580, 50), (580, 150), (450, 150)]},
            {'waypoints': [(50, 350), (580, 350), (580, 50), (50, 50)]}
        ]

    def _update_guards_and_doors(self):
        # Doors: 0=open, 1=open, 2=closed
        door_states = [
            self.special_parts_collected >= 1,
            self.special_parts_collected >= 2,
            False # This one is always a wall
        ]
        self.doors = [d for i, d in enumerate(self._all_doors) if not door_states[i]]
        
        # Guards: Add one guard per special part collected (+1)
        num_guards = min(len(self._guard_definitions), self.special_parts_collected + 1)
        self.guards = []
        for i in range(num_guards):
            defn = self._guard_definitions[i]
            self.guards.append({
                'pos': np.array(defn['waypoints'][0], dtype=float),
                'waypoints': defn['waypoints'],
                'target_idx': 1
            })

    def _spawn_particles(self, pos, base_vel):
        # SFX: Particle fizz
        for _ in range(5):
            angle = (self.np_random.random() - 0.5) * math.pi / 2
            vel = base_vel * self.np_random.uniform(0.5, 1.5)
            rotated_vel = np.array([
                vel[0] * math.cos(angle) - vel[1] * math.sin(angle),
                vel[0] * math.sin(angle) + vel[1] * math.cos(angle)
            ])
            self.particles.append({
                'pos': pos.copy(),
                'vel': rotated_vel,
                'life': self.np_random.uniform(10, 20),
                'max_life': 20
            })
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _is_guard_nearby(self, radius_multiplier=1.5):
        for guard in self.guards:
            dist = np.linalg.norm(self.player_pos - guard['pos'])
            if dist < self.GUARD_DETECTION_RADIUS * radius_multiplier:
                return True
        return False
        
    def _get_dist_to_exit(self, pos):
        return np.linalg.norm(pos - np.array(self.exit_zone.center))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Magnetic Agent")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement_action = 0 # None
        space_action = 0 # Released
        shift_action = 0 # Released
        
        # Poll events for window closing and keyboard
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement_action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement_action = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement_action = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement_action = 4
        
        if keys[pygame.K_SPACE]:
            space_action = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()