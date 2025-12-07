import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:00:10.784102
# Source Brief: brief_00208.md
# Brief Index: 208
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Grow your root through layers of soil, shoot seeds at insects, and switch between planes to clear each level."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to shoot seeds and shift to switch between soil layers."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    FPS = 30

    # Colors
    COLOR_BG = (26, 18, 11)
    SOIL_COLORS = [(51, 32, 18), (69, 45, 29), (87, 60, 43)]
    COLOR_ROOT = (124, 252, 0)
    COLOR_ROOT_GLOW = (124, 252, 0, 50)
    COLOR_INSECT = (255, 69, 0)
    COLOR_INSECT_GLOW = (255, 69, 0, 60)
    COLOR_SEED = (173, 255, 47)
    COLOR_UI_TEXT = (230, 230, 230)
    PARTICLE_COLOR = (139, 69, 19)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.level = 0
        self.game_over = False
        self.win = False

        self.player_pos = None
        self.player_vel = None
        self.player_layer = 0
        self.last_move_dir = None
        self.root_trail = None

        self.insects = None
        self.projectiles = None
        self.particles = None

        self.shoot_cooldown = 0
        self.flip_cooldown = 0
        self.max_seeds = 10
        self.current_seeds = self.max_seeds

        # --- Initialize state variables and validate ---
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # validation is done by the wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Core State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        if options and "level" in options:
            self.level = options["level"]
        else:
            self.level = 1

        # --- Reset Player ---
        self.player_layer = 1
        layer_y_center = self._get_layer_y_center(self.player_layer)
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, layer_y_center)
        self.player_vel = pygame.Vector2(0, 0)
        self.last_move_dir = pygame.Vector2(0, -1) # Default shoot up
        self.root_trail = [self.player_pos.copy() for _ in range(20)]
        self.current_seeds = self.max_seeds

        # --- Reset Entities ---
        self.insects = []
        self.projectiles = []
        self.particles = []
        self._spawn_level()

        # --- Reset Cooldowns ---
        self.shoot_cooldown = 0
        self.flip_cooldown = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        # --- Unpack Action ---
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # --- Pre-action state for reward calculation ---
        dist_before = self._get_distance_to_nearest_insect()

        if not self.game_over:
            # --- Handle Input & Cooldowns ---
            self._handle_input(movement, space_pressed, shift_pressed)
            self.shoot_cooldown = max(0, self.shoot_cooldown - 1)
            self.flip_cooldown = max(0, self.flip_cooldown - 1)

            # --- Update Game Logic ---
            self._update_player()
            reward += self._update_projectiles()
            self._update_insects()
            self._update_particles()
            
            # --- Check Collisions & Calculate Rewards ---
            collision_reward, terminated_by_collision = self._check_player_collision()
            reward += collision_reward
            if terminated_by_collision:
                self.game_over = True
                self.win = False

            dist_after = self._get_distance_to_nearest_insect()
            if dist_after is not None and dist_before is not None:
                if dist_after < dist_before:
                    reward += 0.01 # Small reward for getting closer
                else:
                    reward -= 0.01 # Small penalty for moving away

        # --- Check Termination Conditions ---
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False
        
        if not self.insects and not self.game_over:
            reward += 50
            self.win = True
            terminated = True
            self.level += 1

        if terminated and not self.win:
            reward -= 10 # Penalty for timeout or getting caught

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Internal Update Methods ---

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Movement
        move_dir = pygame.Vector2(0, 0)
        if movement == 1: move_dir.y = -1 # Up
        elif movement == 2: move_dir.y = 1 # Down
        elif movement == 3: move_dir.x = -1 # Left
        elif movement == 4: move_dir.x = 1 # Right
        
        player_speed = 4
        self.player_vel = move_dir * player_speed
        if move_dir.length() > 0:
            self.last_move_dir = move_dir.normalize()

        # Action: Shoot
        if space_pressed and self.shoot_cooldown == 0 and self.current_seeds > 0:
            self._spawn_projectile()
            self.shoot_cooldown = 10 # 1/3 second cooldown
            self.current_seeds -= 1
            # sfx: shoot_sound()

        # Action: Flip Gravity
        if shift_pressed and self.flip_cooldown == 0:
            self._flip_gravity()
            self.flip_cooldown = 15 # 1/2 second cooldown
            # sfx: flip_sound()

    def _update_player(self):
        self.player_pos += self.player_vel
        
        # Clamp to screen and layer boundaries
        layer_height = self.SCREEN_HEIGHT / len(self.SOIL_COLORS)
        layer_top = self.player_layer * layer_height
        layer_bottom = (self.player_layer + 1) * layer_height
        
        self.player_pos.x = np.clip(self.player_pos.x, 10, self.SCREEN_WIDTH - 10)
        self.player_pos.y = np.clip(self.player_pos.y, layer_top + 10, layer_bottom - 10)

        # Update root trail for smooth animation
        self.root_trail.pop(0)
        self.root_trail.append(self.player_pos.copy())
    
    def _update_projectiles(self):
        hit_reward = 0
        for proj in self.projectiles[:]:
            proj['pos'] += proj['vel']
            
            # Check for off-screen
            if not self.screen.get_rect().collidepoint(proj['pos']):
                self.projectiles.remove(proj)
                continue

            # Check for insect collision
            for insect in self.insects[:]:
                if insect['layer'] == proj['layer']:
                    dist = insect['pos'].distance_to(proj['pos'])
                    if dist < 12: # Collision radius
                        self.insects.remove(insect)
                        self.projectiles.remove(proj)
                        self._spawn_particles(insect['pos'], 20)
                        hit_reward += 10 # Reward for hitting an insect
                        # sfx: impact_sound()
                        break
        return hit_reward

    def _update_insects(self):
        for insect in self.insects:
            target_pos = insect['path'][insect['target_idx']]
            direction = (target_pos - insect['pos'])
            
            if direction.length() < insect['speed']:
                insect['target_idx'] = (insect['target_idx'] + 1) % len(insect['path'])
            else:
                insect['pos'] += direction.normalize() * insect['speed']

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_player_collision(self):
        for insect in self.insects:
            if insect['layer'] == self.player_layer:
                dist = self.player_pos.distance_to(insect['pos'])
                if dist < 15: # Player has larger hitbox
                    # sfx: game_over_sound()
                    return -100, True # reward, terminated
        return 0, False

    # --- Spawning and Helper Methods ---

    def _spawn_level(self):
        num_insects = 1 + (self.level - 1) // 2
        insect_speed = 0.5 + 0.05 * ((self.level -1) // 5)
        
        for _ in range(num_insects):
            insect_type = self.np_random.choice(['roamer', 'patroller'])
            layer = self.np_random.integers(0, len(self.SOIL_COLORS))
            layer_y_center = self._get_layer_y_center(layer)
            
            path = []
            if insect_type == 'roamer':
                x1 = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
                x2 = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
                path.append(pygame.Vector2(x1, layer_y_center))
                path.append(pygame.Vector2(x2, layer_y_center))
            else: # Patroller
                cx = self.np_random.uniform(100, self.SCREEN_WIDTH - 100)
                cy = layer_y_center
                w, h = self.np_random.uniform(50, 150), self.np_random.uniform(20, 40)
                path.append(pygame.Vector2(cx - w / 2, cy - h / 2))
                path.append(pygame.Vector2(cx + w / 2, cy - h / 2))
                path.append(pygame.Vector2(cx + w / 2, cy + h / 2))
                path.append(pygame.Vector2(cx - w / 2, cy + h / 2))
            
            self.insects.append({
                'pos': path[0].copy(),
                'layer': layer,
                'speed': insect_speed,
                'path': path,
                'target_idx': 1
            })

    def _spawn_projectile(self):
        proj_speed = 8
        self.projectiles.append({
            'pos': self.player_pos.copy(),
            'vel': self.last_move_dir * proj_speed,
            'layer': self.player_layer
        })

    def _spawn_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 25),
                'radius': self.np_random.uniform(1, 4)
            })
            
    def _flip_gravity(self):
        # Flips between the three layers
        if self.player_layer == 1: # Middle layer can go up or down
            if self.last_move_dir.y < 0: # Moved up last
                self.player_layer = 0
            else:
                self.player_layer = 2
        elif self.player_layer == 0:
            self.player_layer = 1
        elif self.player_layer == 2:
            self.player_layer = 1
        
        # Snap player to the new layer's center
        self.player_pos.y = self._get_layer_y_center(self.player_layer)

    def _get_layer_y_center(self, layer_index):
        layer_height = self.SCREEN_HEIGHT / len(self.SOIL_COLORS)
        return layer_index * layer_height + layer_height / 2

    def _get_distance_to_nearest_insect(self):
        if not self.insects:
            return None
        
        player_insects = [i for i in self.insects if i['layer'] == self.player_layer]
        if not player_insects:
            # If no insects on current layer, find closest on any layer
            distances = [self.player_pos.distance_to(i['pos']) for i in self.insects]
        else:
            distances = [self.player_pos.distance_to(i['pos']) for i in player_insects]
            
        return min(distances) if distances else None

    # --- Rendering Methods ---
    
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        layer_height = self.SCREEN_HEIGHT / len(self.SOIL_COLORS)
        for i, color in enumerate(self.SOIL_COLORS):
            pygame.draw.rect(self.screen, color, (0, i * layer_height, self.SCREEN_WIDTH, layer_height))

    def _render_game(self):
        # Draw root trail
        for i in range(len(self.root_trail) - 1):
            start_pos = self.root_trail[i]
            end_pos = self.root_trail[i+1]
            thickness = int(2 + (i / len(self.root_trail)) * 4)
            alpha = int(50 + (i / len(self.root_trail)) * 150)
            
            # Custom alpha line drawing
            if start_pos.distance_to(end_pos) > 0:
                line_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(line_surf, (*self.COLOR_ROOT, alpha), start_pos, end_pos, thickness)
                self.screen.blit(line_surf, (0,0))
            
        # Draw projectiles
        for proj in self.projectiles:
            if proj['layer'] == self.player_layer:
                pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'].x), int(proj['pos'].y), 4, self.COLOR_SEED)

        # Draw insects
        for insect in self.insects:
            if insect['layer'] == self.player_layer:
                pos_int = (int(insect['pos'].x), int(insect['pos'].y))
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 12, self.COLOR_INSECT_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 8, self.COLOR_INSECT)

        # Draw player
        player_pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], 15, self.COLOR_ROOT_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], 10, self.COLOR_ROOT)
        
        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), self.PARTICLE_COLOR)

    def _render_ui(self):
        # Seeds
        seed_text = self.font_ui.render(f"Seeds: {self.current_seeds}/{self.max_seeds}", True, self.COLOR_UI_TEXT)
        self.screen.blit(seed_text, (10, 10))

        # Level
        level_text = self.font_ui.render(f"Level: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))
        
        # Game Over / Win Message
        if self.game_over and not self.win:
            msg = "CAUGHT!"
        elif self.win:
            msg = "LEVEL CLEAR!"
        else:
            return
            
        text_surf = self.font_game_over.render(msg, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "seeds": self.current_seeds,
            "insects_left": len(self.insects)
        }

    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    env = GameEnv()
    
    # --- Manual Play ---
    # This allows a human to play the game to test the mechanics and visuals
    
    # Key mapping for human control
    # Arrow keys for movement, Space to shoot, Left Shift to flip gravity
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    obs, info = env.reset()
    done = False
    
    # Re-initialize pygame for display window
    pygame.display.init()
    pygame.display.set_caption("Root Runner")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # --- Human Input Processing ---
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
        if done:
            break

        keys = pygame.key.get_pressed()
        for key, move_val in key_to_action.items():
            if keys[key]:
                movement_action = move_val
                break
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        # --- Render to Display ---
        # The observation is already a rendered frame, we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Reset on game over ---
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset(options={"level": info["level"] if info.get('insects_left', 1) == 0 else 1})
        
        clock.tick(GameEnv.FPS)

    env.close()