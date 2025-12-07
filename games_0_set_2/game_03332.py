
# Generated: 2025-08-27T23:02:44.925061
# Source Brief: brief_03332.md
# Brief Index: 3332

        
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


# Helper class for floating text and particles
class Effect:
    """A class for visual effects like particles and floating text."""
    def __init__(self, x, y, text=None, color=(255, 255, 255), size=20, life=30, vel=(0, -1)):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life
        self.vel_x = vel[0]
        self.vel_y = vel[1]

    def update(self):
        """Updates the effect's position and lifetime."""
        self.x += self.vel_x
        self.y += self.vel_y
        self.life -= 1

    def draw(self, surface, font):
        """Draws the effect on the given surface."""
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            if self.text:
                text_surf = font.render(self.text, True, self.color)
                text_surf.set_alpha(alpha)
                surface.blit(text_surf, (self.x, self.y))
            else: # Particle
                radius = int(self.size * (self.life / self.max_life))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), radius, (*self.color, alpha))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move. Enter combat by walking into an enemy. "
        "In combat: Space to attack, Shift to use a potion, or arrow keys to flee."
    )

    game_description = (
        "A turn-based dungeon crawler. Navigate through rooms, defeat enemies, "
        "and manage your health to reach the final room."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.TILE_SIZE = 20
        self.MAX_STEPS = 1000
        self.MAX_ROOMS = 5
        self.HERO_MAX_HEALTH = 3

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_WALL = (80, 80, 90)
        self.COLOR_DOOR = (139, 69, 19)
        self.COLOR_HERO = (0, 255, 127) # Bright Green
        self.COLOR_ENEMY = (255, 69, 0) # Bright Red
        self.COLOR_POTION = (30, 144, 255) # Bright Blue
        self.COLOR_UI_BG = (40, 40, 55)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_HEALTH_BAR = (0, 200, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_room = 1
        self.hero_health = self.HERO_MAX_HEALTH
        self.num_potions = 0
        self.hero_pos = (0, 0)
        self.door_pos = (0, 0)
        self.enemies = []
        self.combat_active = False
        self.combat_enemy_idx = -1
        self.effects = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_room = 1
        self.hero_health = self.HERO_MAX_HEALTH
        self.num_potions = 1
        self.combat_active = False
        self.combat_enemy_idx = -1
        self.effects = []

        self._setup_room()

        return self._get_observation(), self._get_info()

    def _setup_room(self):
        """Initializes the hero, enemies, and door for the current room."""
        self.enemies = []
        num_enemies = self.current_room
        
        # Define valid spawn area (1 tile away from edges and UI)
        valid_spawns = []
        for x in range(1, self.GRID_W - 1):
            for y in range(1, self.GRID_H - 2):
                valid_spawns.append((x, y))

        # Place door
        self.door_pos = (self.GRID_W // 2, 0)
        if self.door_pos in valid_spawns: valid_spawns.remove(self.door_pos)

        # Place hero
        self.hero_pos = (self.GRID_W // 2, self.GRID_H - 3)
        if self.hero_pos in valid_spawns: valid_spawns.remove(self.hero_pos)

        # Place enemies
        num_to_spawn = min(num_enemies, len(valid_spawns))
        enemy_indices = self.np_random.choice(len(valid_spawns), num_to_spawn, replace=False)
        self.enemies = [valid_spawns[i] for i in enemy_indices]
        
    def _add_effect(self, effect):
        """Adds a visual effect to the effects list."""
        self.effects.append(effect)

    def _create_particles(self, pos, count, color, speed_range=(1, 3)):
        """Spawns a number of particles at a given grid position."""
        px, py = self._grid_to_pixel(pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(speed_range[0], speed_range[1])
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(15, 30)
            size = self.np_random.integers(3, 7)
            self._add_effect(Effect(px, py, color=color, size=size, life=life, vel=vel))

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.game_over = False

        if self.combat_active:
            reward += self._handle_combat_action(movement, space_pressed, shift_pressed)
        else:
            reward += self._handle_explore_action(movement, shift_pressed)
        
        self.score += reward
        self.steps += 1
        
        # Check termination conditions
        terminated = False
        if self.hero_health <= 0:
            if not self.game_over: # Add terminal penalty only once
                reward -= 100
                self.score -= 100
            terminated = True
            self.game_over = True
        elif self.current_room > self.MAX_ROOMS:
            if not self.game_over: # Add terminal reward only once
                reward += 100
                self.score += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_combat_action(self, movement, space_pressed, shift_pressed):
        """Processes player actions during combat."""
        reward = 0
        # Action priority: Attack > Potion > Flee
        if space_pressed: # Attack
            # Sound: Attack Hit
            reward += 5
            px, py = self._grid_to_pixel(self.enemies[self.combat_enemy_idx])
            self._add_effect(Effect(px - 15, py - 30, text="+5", color=(255, 255, 0)))
            self._create_particles(self.enemies[self.combat_enemy_idx], 20, self.COLOR_ENEMY)
            
            del self.enemies[self.combat_enemy_idx]
            self.combat_active = False
            self.combat_enemy_idx = -1
        
        elif shift_pressed: # Use Potion
            reward += self._use_potion()
        
        elif movement > 0: # Flee
            # Sound: Flee
            if self.np_random.random() < 0.5: # 50% chance to take damage
                self.hero_health -= 1
                reward -= 1
                px, py = self._grid_to_pixel(self.hero_pos)
                self._add_effect(Effect(px - 15, py - 30, text="-1 HP", color=(255, 0, 0)))
                self._create_particles(self.hero_pos, 15, (200, 0, 0))
            self.combat_active = False
            self.combat_enemy_idx = -1

        return reward

    def _handle_explore_action(self, movement, shift_pressed):
        """Processes player actions during exploration."""
        reward = 0
        if shift_pressed:
            reward += self._use_potion()
        
        if movement > 0:
            old_pos = self.hero_pos
            old_dist_to_door = abs(old_pos[0] - self.door_pos[0]) + abs(old_pos[1] - self.door_pos[1])
            
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1 # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1 # Right

            new_pos = (self.hero_pos[0] + dx, self.hero_pos[1] + dy)

            # Check boundaries
            if 0 <= new_pos[0] < self.GRID_W and 0 <= new_pos[1] < self.GRID_H - 2:
                self.hero_pos = new_pos
            
            # Check for door
            if self.hero_pos == self.door_pos:
                # Sound: Next Level
                self.current_room += 1
                if self.current_room <= self.MAX_ROOMS:
                    self._setup_room()
                return reward # Return early to avoid other checks

            # Check for enemy collision
            for i, enemy_pos in enumerate(self.enemies):
                if self.hero_pos == enemy_pos:
                    # Sound: Encounter
                    self.combat_active = True
                    self.combat_enemy_idx = i
                    self.hero_health -= 1
                    reward -= 1
                    px, py = self._grid_to_pixel(self.hero_pos)
                    self._add_effect(Effect(px - 15, py - 30, text="-1 HP", color=(255, 0, 0)))
                    self._create_particles(self.hero_pos, 15, (200, 0, 0))
                    self.hero_pos = old_pos # Move hero back
                    return reward

            # Movement reward
            new_dist_to_door = abs(self.hero_pos[0] - self.door_pos[0]) + abs(self.hero_pos[1] - self.door_pos[1])
            if new_dist_to_door < old_dist_to_door:
                reward += 0.1
            elif new_dist_to_door > old_dist_to_door:
                reward -= 0.2

        return reward

    def _use_potion(self):
        """Uses a potion if available and hero is not at max health."""
        if self.num_potions > 0 and self.hero_health < self.HERO_MAX_HEALTH:
            # Sound: Potion Drink
            self.num_potions -= 1
            self.hero_health += 1
            assert self.hero_health <= self.HERO_MAX_HEALTH
            px, py = self._grid_to_pixel(self.hero_pos)
            self._add_effect(Effect(px - 15, py - 30, text="+2", color=(0, 255, 255)))
            self._create_particles(self.hero_pos, 20, self.COLOR_POTION)
            return 2
        return 0

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to pixel coordinates (center of tile)."""
        x = grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2
        y = grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        return int(x), int(y)

    def _get_observation(self):
        """Renders the game state to a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_render_effects()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the main game elements (grid, entities)."""
        # Draw grid
        for x in range(0, self.WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw door
        dx, dy = self._grid_to_pixel(self.door_pos)
        pygame.draw.rect(self.screen, self.COLOR_DOOR, (dx - self.TILE_SIZE//2, dy - self.TILE_SIZE//2, self.TILE_SIZE, self.TILE_SIZE))

        # Draw enemies
        for i, enemy_pos in enumerate(self.enemies):
            ex, ey = self._grid_to_pixel(enemy_pos)
            color = (255, 165, 0) if self.combat_active and i == self.combat_enemy_idx else self.COLOR_ENEMY
            pygame.gfxdraw.filled_circle(self.screen, ex, ey, self.TILE_SIZE // 2 - 2, color)
            pygame.gfxdraw.aacircle(self.screen, ex, ey, self.TILE_SIZE // 2 - 2, color)

        # Draw hero
        hx, hy = self._grid_to_pixel(self.hero_pos)
        pygame.gfxdraw.filled_circle(self.screen, hx, hy, self.TILE_SIZE // 2 - 2, self.COLOR_HERO)
        pygame.gfxdraw.aacircle(self.screen, hx, hy, self.TILE_SIZE // 2 - 2, self.COLOR_HERO)
        pygame.gfxdraw.aacircle(self.screen, hx, hy, self.TILE_SIZE // 2, (*self.COLOR_HERO, 100)) # Glow

    def _update_and_render_effects(self):
        """Updates and renders all active visual effects."""
        self.effects = [e for e in self.effects if e.life > 0]
        for effect in self.effects:
            effect.update()
            effect.draw(self.screen, self.font_s)

    def _render_ui(self):
        """Renders the user interface overlay."""
        ui_y = self.HEIGHT - self.TILE_SIZE * 2
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, ui_y, self.WIDTH, self.TILE_SIZE * 2))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, ui_y), (self.WIDTH, ui_y), 2)

        # Room text
        room_text = self.font_m.render(f"Room: {min(self.current_room, self.MAX_ROOMS)}/{self.MAX_ROOMS}", True, self.COLOR_TEXT)
        self.screen.blit(room_text, (10, ui_y + 5))
        
        # Score text
        score_text = self.font_m.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, ui_y + 5))

        # Health
        hp_text = self.font_m.render(f"HP: {self.hero_health}/{self.HERO_MAX_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(hp_text, (10, ui_y + 35))
        
        # Health bar
        bar_x, bar_y, bar_w, bar_h = 90, ui_y + 38, 100, 15
        health_ratio = max(0, self.hero_health / self.HERO_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_w * health_ratio), bar_h))
        
        # Potions
        potion_text = self.font_m.render(f"Potions: {self.num_potions}", True, self.COLOR_TEXT)
        self.screen.blit(potion_text, (210, ui_y + 35))

        # Combat text
        if self.combat_active:
            combat_title = self.font_l.render("COMBAT!", True, self.COLOR_ENEMY)
            self.screen.blit(combat_title, (self.WIDTH // 2 - combat_title.get_width() // 2, 20))
            actions_text = self.font_s.render("Space: Attack | Shift: Potion | Arrows: Flee", True, self.COLOR_TEXT)
            self.screen.blit(actions_text, (self.WIDTH // 2 - actions_text.get_width() // 2, 70))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.current_room > self.MAX_ROOMS else "GAME OVER"
            color = (0, 255, 0) if self.current_room > self.MAX_ROOMS else (255, 0, 0)
            end_text = self.font_l.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "room": self.current_room,
            "health": self.hero_health,
            "potions": self.num_potions,
            "combat": self.combat_active,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    total_reward = 0
    
    print(env.user_guide)

    last_action_processed = True
    while not done:
        movement, space, shift = 0, 0, 0
        should_step = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                should_step = True
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]: movement = 1
                elif keys[pygame.K_DOWN]: movement = 2
                elif keys[pygame.K_LEFT]: movement = 3
                elif keys[pygame.K_RIGHT]: movement = 4
                if keys[pygame.K_SPACE]: space = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        if not should_step and not last_action_processed:
            should_step = True # Process a no-op to advance state
        
        if should_step:
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total: {total_reward:.2f}, Info: {info}")
            done = terminated or truncated
            last_action_processed = (action != [0,0,0])

        # Render to Screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    
    end_time = pygame.time.get_ticks() + 3000
    while pygame.time.get_ticks() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: break
        pygame.display.flip()
        env.clock.tick(30)

    env.close()