import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:21:21.105838
# Source Brief: brief_00363.md
# Brief Index: 363
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
        "Infiltrate a secure facility as a size-shifting agent. Evade guards by shrinking, grow to fight the boss, and steal the secret documents to win."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to shrink or grow. Use shift to attack the boss when you are large and nearby."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_FPS = 30
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (40, 50, 80)
    COLOR_FURNITURE = (50, 65, 100)
    COLOR_PLAYER_LARGE = (255, 120, 0) # Bright Orange
    COLOR_PLAYER_SMALL = (0, 255, 150) # Bright Green
    COLOR_GUARD = (255, 50, 50)
    COLOR_GUARD_VISION = (100, 20, 20, 80)
    COLOR_BOSS = (180, 0, 255)
    COLOR_DOCS = (255, 220, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 40, 60, 180)

    # Player
    PLAYER_SPEED = 4.0
    PLAYER_SIZE_LARGE = 12
    PLAYER_SIZE_SMALL = 5
    PLAYER_SIZE_RATE = 0.2

    # Guard
    GUARD_SIZE = 10
    GUARD_VISION_RADIUS = 80
    GUARD_BASE_SPEED = 1.0

    # Boss
    BOSS_SIZE = 25
    BOSS_BASE_HEALTH = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gym Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Persistent State (Difficulty Scaling) ---
        self.boss_health_modifier = 1.0
        self.guard_speed_modifier = 0.0

        # --- Initialize State Variables ---
        self.player_pos = None
        self.player_size = None
        self.player_target_size = None
        self.guards = None
        self.walls = None
        self.boss = None
        self.docs = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.last_space_state = 0
        self.last_shift_state = 0
        self.reward_log = []

        self.reset()
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.last_space_state = 0
        self.last_shift_state = 0
        self.reward_log = []

        self._generate_layout()

        self.player_pos = np.array([50.0, self.SCREEN_HEIGHT / 2.0])
        self.player_size = float(self.PLAYER_SIZE_LARGE)
        self.player_target_size = float(self.PLAYER_SIZE_LARGE)

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement, space_action, shift_action = action
        space_pressed = space_action == 1 and self.last_space_state == 0
        shift_pressed = shift_action == 1 and self.last_shift_state == 0
        self.last_space_state = space_action
        self.last_shift_state = shift_action
        
        reward = 0
        
        if not self.game_over and not self.game_won:
            # --- Update Game Logic ---
            self._handle_input(movement, space_pressed, shift_pressed)
            self._update_player()
            self._update_guards()
            self._update_boss()
            self._update_particles()
            
            # --- Check Events and Calculate Rewards ---
            event_reward = self._check_events()
            reward += event_reward

            # Continuous rewards
            is_large = self.player_size > (self.PLAYER_SIZE_SMALL + 1)
            is_near_guard = any(np.linalg.norm(self.player_pos - g['pos']) < self.GUARD_VISION_RADIUS for g in self.guards)
            
            if is_large and is_near_guard:
                reward -= 0.5 # Penalty for being large near a guard
            elif not is_large:
                reward += 0.1 # Reward for being small and stealthy
        
        self.score += reward
        self.steps += 1
        
        # --- Check Termination ---
        terminated = self.game_over or self.game_won
        truncated = self.steps >= self.MAX_STEPS

        if self.game_over:
            reward -= 100
            self.score -= 100
            # Increase difficulty for next run
            self.boss_health_modifier += 0.2
            self.guard_speed_modifier += 0.05
        elif self.game_won:
            reward += 100
            self.score += 100
            # Reset difficulty on win
            self.boss_health_modifier = 1.0
            self.guard_speed_modifier = 0.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Movement
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] = -1.0 # Up
        elif movement == 2: move_vec[1] = 1.0  # Down
        elif movement == 3: move_vec[0] = -1.0 # Left
        elif movement == 4: move_vec[0] = 1.0  # Right
        
        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec)
        
        new_pos = self.player_pos + move_vec * self.PLAYER_SPEED
        
        # Wall collision
        player_rect = pygame.Rect(new_pos[0] - self.player_size, new_pos[1] - self.player_size, self.player_size * 2, self.player_size * 2)
        if not any(wall.colliderect(player_rect) for wall in self.walls):
             self.player_pos = new_pos

        # Boundary collision
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_size, self.SCREEN_WIDTH - self.player_size)
        self.player_pos[1] = np.clip(self.player_pos[1], self.player_size, self.SCREEN_HEIGHT - self.player_size)

        # Shrink/Grow Toggle (Spacebar)
        if space_pressed:
            # SFX: Play shrink/grow sound
            if self.player_target_size == self.PLAYER_SIZE_LARGE:
                self.player_target_size = self.PLAYER_SIZE_SMALL
            else:
                self.player_target_size = self.PLAYER_SIZE_LARGE
            
            # Particle effect for size change
            for _ in range(30):
                self._create_particle(self.player_pos, self.COLOR_PLAYER_LARGE if self.player_target_size == self.PLAYER_SIZE_LARGE else self.COLOR_PLAYER_SMALL, 2, 4, 20, 40)

        # Gadget Use (Shift)
        if shift_pressed and self.boss['active']:
             # In this brief, the only "gadget" is attacking the boss
            if np.linalg.norm(self.player_pos - self.boss['pos']) < self.boss['size'] + self.player_size + 10 and self.player_size > self.PLAYER_SIZE_SMALL + 1:
                # SFX: Play attack sound
                self.boss['health'] -= 10
                self.reward_log.append(("+5", self.steps))
                # Particle effect for hit
                for _ in range(50):
                    self._create_particle(self.boss['pos'], self.COLOR_BOSS, 3, 6, 30, 60)


    def _update_player(self):
        # Smoothly interpolate player size
        if self.player_size != self.player_target_size:
            diff = self.player_target_size - self.player_size
            self.player_size += np.sign(diff) * min(abs(diff), self.PLAYER_SIZE_RATE)

    def _update_guards(self):
        for guard in self.guards:
            target_node_idx = guard['target_node']
            target_pos = guard['path'][target_node_idx]
            
            direction = target_pos - guard['pos']
            distance = np.linalg.norm(direction)
            
            if distance < 2:
                guard['target_node'] = (target_node_idx + 1) % len(guard['path'])
            else:
                guard['pos'] += (direction / distance) * (self.GUARD_BASE_SPEED + self.guard_speed_modifier)

    def _update_boss(self):
        if np.linalg.norm(self.player_pos - self.boss['pos']) < 100 and self.player_size > self.PLAYER_SIZE_SMALL + 1:
            self.boss['active'] = True
        
        if self.boss['active'] and self.boss['health'] <= 0:
            self.docs['stolen'] = False # Documents are now available

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['base_size'] * (p['life'] / p['max_life']))

    def _check_events(self):
        reward = 0
        
        # Guard detection
        is_large = self.player_size > self.PLAYER_SIZE_SMALL + 1
        if is_large:
            for guard in self.guards:
                if np.linalg.norm(self.player_pos - guard['pos']) < self.GUARD_VISION_RADIUS:
                    self.game_over = True
                    self.reward_log.append(("-100 CAUGHT", self.steps))
                    # SFX: Play detection/alarm sound
                    return 0 # Terminal reward handled in step()

        # Boss defeat
        if self.boss['health'] <= 0 and not self.docs['stolen']:
             # Player still needs to grab the documents
             pass

        # Document theft (Win condition)
        if self.boss['health'] <= 0 and not self.docs['stolen']:
            if np.linalg.norm(self.player_pos - self.docs['pos']) < self.player_size + 5:
                self.docs['stolen'] = True
                self.game_won = True
                self.reward_log.append(("+100 WIN", self.steps))
                # SFX: Play victory sound
                return 0 # Terminal reward handled in step()
        
        return reward

    def _generate_layout(self):
        self.walls = []
        # Outer bounds
        self.walls.append(pygame.Rect(0, 0, self.SCREEN_WIDTH, 10))
        self.walls.append(pygame.Rect(0, self.SCREEN_HEIGHT - 10, self.SCREEN_WIDTH, 10))
        self.walls.append(pygame.Rect(0, 0, 10, self.SCREEN_HEIGHT))
        self.walls.append(pygame.Rect(self.SCREEN_WIDTH - 10, 0, 10, self.SCREEN_HEIGHT))

        # Central structure
        self.walls.append(pygame.Rect(150, 100, 10, 200))
        self.walls.append(pygame.Rect(150, 100, 340, 10))
        self.walls.append(pygame.Rect(150, 290, 340, 10))
        
        # Boss room wall
        self.walls.append(pygame.Rect(490, 10, 10, self.SCREEN_HEIGHT - 20))


        self.guards = [
            {
                'pos': np.array([200.0, 50.0]),
                'path': [np.array([200.0, 50.0]), np.array([450.0, 50.0])],
                'target_node': 1
            },
            {
                'pos': np.array([450.0, 350.0]),
                'path': [np.array([450.0, 350.0]), np.array([200.0, 350.0])],
                'target_node': 1
            }
        ]

        boss_pos = np.array([self.SCREEN_WIDTH - 60.0, self.SCREEN_HEIGHT / 2.0])
        self.boss = {
            'pos': boss_pos,
            'health': self.BOSS_BASE_HEALTH * self.boss_health_modifier,
            'max_health': self.BOSS_BASE_HEALTH * self.boss_health_modifier,
            'active': False,
            'size': self.BOSS_SIZE
        }
        self.docs = {
            'pos': boss_pos.copy(),
            'stolen': True # Can't be stolen until boss is defeated
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Render guard vision cones
        for guard in self.guards:
            vision_surface = pygame.Surface((self.GUARD_VISION_RADIUS * 2, self.GUARD_VISION_RADIUS * 2), pygame.SRCALPHA)
            pygame.draw.circle(vision_surface, self.COLOR_GUARD_VISION, (self.GUARD_VISION_RADIUS, self.GUARD_VISION_RADIUS), self.GUARD_VISION_RADIUS)
            self.screen.blit(vision_surface, (int(guard['pos'][0] - self.GUARD_VISION_RADIUS), int(guard['pos'][1] - self.GUARD_VISION_RADIUS)))

        # Render particles
        self._render_particles()

        # Render guards
        for guard in self.guards:
            self._draw_glowing_circle(self.screen, self.COLOR_GUARD, guard['pos'], self.GUARD_SIZE, 5)

        # Render boss and documents
        if self.boss['health'] > 0:
            self._draw_glowing_circle(self.screen, self.COLOR_BOSS, self.boss['pos'], self.boss['size'], 10)
        elif not self.docs['stolen']:
            self._draw_glowing_circle(self.screen, self.COLOR_DOCS, self.docs['pos'], 10, 20)

        # Render player
        is_large = self.player_size > self.PLAYER_SIZE_SMALL + 1
        player_color = self.COLOR_PLAYER_LARGE if is_large else self.COLOR_PLAYER_SMALL
        self._draw_glowing_circle(self.screen, player_color, self.player_pos, self.player_size, 10)

    def _render_ui(self):
        # --- UI Background Panel ---
        panel_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 40, self.SCREEN_WIDTH, 40)
        s = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (0, self.SCREEN_HEIGHT - 40))

        # --- Score and Steps ---
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, self.SCREEN_HEIGHT - 30))
        self.screen.blit(steps_text, (150, self.SCREEN_HEIGHT - 30))
        
        # --- Player Size Indicator ---
        size_ratio = (self.player_size - self.PLAYER_SIZE_SMALL) / (self.PLAYER_SIZE_LARGE - self.PLAYER_SIZE_SMALL)
        size_text = self.font_small.render("SIZE:", True, self.COLOR_UI_TEXT)
        self.screen.blit(size_text, (self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 30))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - 100, self.SCREEN_HEIGHT - 28, 80, 14))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_LARGE, (self.SCREEN_WIDTH - 98, self.SCREEN_HEIGHT - 26, 76 * size_ratio, 10))

        # --- Boss Health Bar ---
        if self.boss['active']:
            health_ratio = self.boss['health'] / self.boss['max_health']
            bar_width = 300
            bar_x = (self.SCREEN_WIDTH - bar_width) / 2
            pygame.draw.rect(self.screen, (0,0,0), (bar_x - 2, 18, bar_width + 4, 24))
            pygame.draw.rect(self.screen, self.COLOR_GUARD, (bar_x, 20, bar_width, 20))
            if health_ratio > 0:
                pygame.draw.rect(self.screen, self.COLOR_BOSS, (bar_x, 20, bar_width * health_ratio, 20))
            boss_text = self.font_small.render("FINAL BOSS", True, self.COLOR_UI_TEXT)
            self.screen.blit(boss_text, (bar_x + bar_width/2 - boss_text.get_width()/2, 22))

        # --- Game Over/Win Message ---
        if self.game_over:
            self._draw_overlay_message("CAUGHT!")
        elif self.game_won:
            self._draw_overlay_message("DOCUMENTS STOLEN!")
            
    def _draw_overlay_message(self, message):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 128))
        self.screen.blit(s, (0, 0))
        text_surf = self.font_large.render(message, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def _draw_glowing_circle(self, surface, color, pos, radius, glow_strength):
        x, y = int(pos[0]), int(pos[1])
        r, g, b = color
        
        # Draw glow
        for i in range(glow_strength):
            alpha = 100 * (1 - i / glow_strength)**2
            pygame.gfxdraw.filled_circle(surface, x, y, int(radius + i), (r, g, b, alpha))

        # Draw main circle
        pygame.gfxdraw.aacircle(surface, x, y, int(radius), color)
        pygame.gfxdraw.filled_circle(surface, x, y, int(radius), color)

    def _create_particle(self, pos, color, min_speed, max_speed, min_life, max_life):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(min_speed, max_speed)
        life = random.randint(min_life, max_life)
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed
        self.particles.append({
            'pos': pos.copy(),
            'vel': vel,
            'life': life,
            'max_life': life,
            'color': color,
            'base_size': random.uniform(1, 4)
        })

    def _render_particles(self):
        for p in self.particles:
            r, g, b = p['color']
            alpha = 255 * (p['life'] / p['max_life'])
            color = (r, g, b, alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_size": self.player_size,
            "boss_health": self.boss['health'],
            "is_large": self.player_size > self.PLAYER_SIZE_SMALL + 1,
            "game_won": self.game_won
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will create a window and let you control the agent
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Manual Test: Shrink Agent")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    print("\n" + "="*30)
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("R: Reset, Q: Quit")
    print("="*30 + "\n")
    
    while True:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    truncated = False

        if terminated or truncated:
            # Just display the final frame until reset
            pass
        else:
            # --- Action Mapping from Keyboard ---
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Step Environment ---
            obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.GAME_FPS)