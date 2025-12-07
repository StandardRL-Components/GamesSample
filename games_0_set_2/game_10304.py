import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:09:32.251963
# Source Brief: brief_00304.md
# Brief Index: 304
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a size-shifting character.
    The goal is to defeat 15 enemies by punching them while collecting energy orbs
    to maintain and increase size. Contact with enemies shrinks the player.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a size-shifting character, collecting orbs to grow and punching enemies to win. "
        "Avoid contact with enemies, as it will shrink you."
    )
    user_guide = "Controls: ←→ to move left and right. Press space to punch when your size is large enough."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.WIN_CONDITION_ENEMIES = 15
        
        # Colors (Bright, high-contrast)
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ORB = (255, 220, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PUNCH = (255, 255, 255)

        # Player settings
        self.PLAYER_BASE_RADIUS = 15
        self.PLAYER_SPEED = 8.0
        self.PLAYER_PUNCH_SIZE_REQ = 150.0
        self.PLAYER_PUNCH_RADIUS = 50
        self.PLAYER_PUNCH_COOLDOWN = 15 # steps
        
        # Enemy settings
        self.ENEMY_RADIUS = 10
        self.ENEMY_SPEED = 1.5
        self.ENEMY_SPAWN_INTERVAL = 50
        self.ENEMY_DIFFICULTY_INTERVAL = 500

        # Orb settings
        self.ORB_RADIUS = 7
        self.MAX_ORBS = 5

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_player = pygame.font.Font(None, 22)

        # --- Internal State Attributes ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = None
        self.player_size = 0.0 # Percentage
        self.punch_cooldown_timer = 0
        self.space_was_held = False

        self.enemies = []
        self.orbs = []
        self.particles = []
        
        self.enemies_defeated = 0
        self.enemy_spawn_timer = 0
        self.enemies_to_spawn = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_size = 100.0
        self.punch_cooldown_timer = 0
        self.space_was_held = False

        self.enemies = []
        self.orbs = []
        self.particles = []
        
        self.enemies_defeated = 0
        self.enemy_spawn_timer = self.ENEMY_SPAWN_INTERVAL
        self.enemies_to_spawn = 1

        for _ in range(self.MAX_ORBS):
            self._spawn_orb()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        punch_triggered = space_held and not self.space_was_held and self.punch_cooldown_timer == 0

        self._handle_player_movement(movement)
        
        if punch_triggered:
            reward += self._handle_punch()
        
        self.space_was_held = space_held

        # --- Update Game Logic ---
        self._update_timers()
        self._update_enemies()
        self._update_particles()
        self._spawn_enemies()

        # --- Collision Detection ---
        reward += self._handle_orb_collisions()
        reward += self._handle_enemy_collisions()
        
        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.player_size <= 0:
            reward -= 100
            terminated = True
        elif self.enemies_defeated >= self.WIN_CONDITION_ENEMIES:
            reward += 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True

        self.game_over = terminated or truncated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Internal Logic Methods ---

    def _handle_player_movement(self, movement):
        if movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        current_radius = self._get_player_radius()
        self.player_pos.x = max(current_radius, min(self.WIDTH - current_radius, self.player_pos.x))
        self.player_pos.y = self.HEIGHT / 2 # Player is fixed on y-axis

    def _handle_punch(self):
        reward = 0
        self.punch_cooldown_timer = self.PLAYER_PUNCH_COOLDOWN
        
        if self.player_size > self.PLAYER_PUNCH_SIZE_REQ:
            # Sound: Punch Whoosh
            self._spawn_particles(self.player_pos, 30, self.COLOR_PUNCH, 8, 20, 1)
            
            enemies_hit = []
            for enemy in self.enemies:
                if self.player_pos.distance_to(enemy['pos']) < self.PLAYER_PUNCH_RADIUS + self.ENEMY_RADIUS:
                    enemies_hit.append(enemy)
            
            for enemy in enemies_hit:
                self.enemies.remove(enemy)
                self.enemies_defeated += 1
                reward += 5.0
                # Sound: Enemy Explosion
                self._spawn_particles(enemy['pos'], 20, self.COLOR_ENEMY, 6, 15)
        else:
            # Sound: Punch Fail
            self._spawn_particles(self.player_pos, 5, (100,100,100), 2, 10, 0.5)
            
        return reward

    def _update_timers(self):
        self.punch_cooldown_timer = max(0, self.punch_cooldown_timer - 1)
        self.enemy_spawn_timer = max(0, self.enemy_spawn_timer - 1)
        if self.steps > 0 and self.steps % self.ENEMY_DIFFICULTY_INTERVAL == 0:
            self.enemies_to_spawn += 1

    def _update_enemies(self):
        for enemy in self.enemies:
            direction = (self.player_pos - enemy['pos']).normalize()
            enemy['pos'] += direction * self.ENEMY_SPEED

    def _update_particles(self):
        # Iterate backwards to safely remove items
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.pop(i)

    def _spawn_enemies(self):
        if self.enemy_spawn_timer == 0:
            for _ in range(self.enemies_to_spawn):
                spawn_x = 0 if random.random() < 0.5 else self.WIDTH
                spawn_y = random.uniform(self.ENEMY_RADIUS, self.HEIGHT - self.ENEMY_RADIUS)
                self.enemies.append({'pos': pygame.Vector2(spawn_x, spawn_y)})
            self.enemy_spawn_timer = self.ENEMY_SPAWN_INTERVAL
    
    def _spawn_orb(self):
        pos = pygame.Vector2(
            random.uniform(self.ORB_RADIUS, self.WIDTH - self.ORB_RADIUS),
            random.uniform(self.ORB_RADIUS, self.HEIGHT - self.ORB_RADIUS)
        )
        self.orbs.append({'pos': pos})

    def _handle_orb_collisions(self):
        reward = 0
        player_radius = self._get_player_radius()
        orbs_collected = []
        for orb in self.orbs:
            if self.player_pos.distance_to(orb['pos']) < player_radius + self.ORB_RADIUS:
                orbs_collected.append(orb)
                self.player_size += 20.0
                reward += 0.1
                # Sound: Orb Collect
                self._spawn_particles(orb['pos'], 10, self.COLOR_ORB, 3, 10)
        
        if orbs_collected:
            self.orbs = [o for o in self.orbs if o not in orbs_collected]
            for _ in range(len(orbs_collected)):
                self._spawn_orb()
        return reward

    def _handle_enemy_collisions(self):
        reward = 0
        player_radius = self._get_player_radius()
        enemies_collided = []
        for enemy in self.enemies:
            if self.player_pos.distance_to(enemy['pos']) < player_radius + self.ENEMY_RADIUS:
                enemies_collided.append(enemy)
                self.player_size -= 30.0
                reward -= 0.5
                # Sound: Player Hit
                self._spawn_particles(self.player_pos, 15, self.COLOR_PLAYER, 4, 12)

        if enemies_collided:
            self.enemies = [e for e in self.enemies if e not in enemies_collided]
            for enemy in enemies_collided: # Spawn particles for each collided enemy
                self._spawn_particles(enemy['pos'], 10, self.COLOR_ENEMY, 2, 8)
        return reward

    # --- Helper and Rendering Methods ---

    def _get_player_radius(self):
        return self.PLAYER_BASE_RADIUS * (self.player_size / 100.0)

    def _spawn_particles(self, pos, count, color, speed, lifespan, radius_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel_magnitude = random.uniform(speed * 0.5, speed)
            velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * vel_magnitude
            self.particles.append({
                'pos': pos.copy(),
                'vel': velocity,
                'lifespan': random.randint(lifespan // 2, lifespan),
                'color': color,
                'radius': random.uniform(1, 3) * radius_mult
            })
            
    def _draw_glowing_circle(self, surface, pos, radius, color, glow_layers=5, glow_alpha=30):
        if radius <= 0: return
        x, y = int(pos.x), int(pos.y)
        
        # Draw glow
        for i in range(glow_layers, 0, -1):
            glow_radius = int(radius + i * 2)
            glow_color = (*color, glow_alpha)
            pygame.gfxdraw.filled_circle(surface, x, y, glow_radius, glow_color)
            pygame.gfxdraw.aacircle(surface, x, y, glow_radius, glow_color)

        # Draw main circle
        pygame.gfxdraw.filled_circle(surface, x, y, int(radius), color)
        pygame.gfxdraw.aacircle(surface, x, y, int(radius), color)

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
            "player_size": self.player_size,
            "enemies_defeated": self.enemies_defeated
        }
        
    def _render_game(self):
        # Draw background grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], p['radius'])

        # Draw orbs
        pulse = (math.sin(self.steps * 0.15) + 1) / 2 * 5 + 1 # Pulsating glow size
        for orb in self.orbs:
            self._draw_glowing_circle(self.screen, orb['pos'], self.ORB_RADIUS, self.COLOR_ORB, glow_layers=3, glow_alpha=int(pulse*10))

        # Draw enemies
        for enemy in self.enemies:
            self._draw_glowing_circle(self.screen, enemy['pos'], self.ENEMY_RADIUS, self.COLOR_ENEMY)

        # Draw player
        player_radius = max(0, self._get_player_radius())
        self._draw_glowing_circle(self.screen, self.player_pos, player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Player size text
        player_radius = self._get_player_radius()
        size_text = f"{int(self.player_size)}%"
        text_surf = self.font_player.render(size_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.player_pos.x, self.player_pos.y - player_radius - 15))
        self.screen.blit(text_surf, text_rect)

        # Top-right UI
        enemy_text = f"Enemies Defeated: {self.enemies_defeated} / {self.WIN_CONDITION_ENEMIES}"
        text_surf = self.font_ui.render(enemy_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))
        
        # Punch ready indicator
        if self.player_size > self.PLAYER_PUNCH_SIZE_REQ:
            color = self.COLOR_PUNCH if self.punch_cooldown_timer == 0 else (100, 100, 100)
            punch_text = "PUNCH READY"
            text_surf = self.font_ui.render(punch_text, True, color)
            self.screen.blit(text_surf, (10, 10))

    def close(self):
        pygame.quit()

# Example usage for testing and visualization
if __name__ == "__main__":
    # This block will not run in the testing environment, but is useful for local development.
    # To run, you'll need to `pip install pygame`.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    done = False
    
    # Override Pygame screen for direct display
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Size Shifter")
    clock = pygame.time.Clock()

    action = [0, 0, 0] # [movement, space, shift]
    
    print("\n--- Manual Control ---")
    print("A/D or Left/Right Arrow: Move")
    print("Spacebar: Punch")
    print("Q or ESC: Quit")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    done = True
            # This logic is for press-and-release punch
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_SPACE:
            #         action[1] = 1
            # if event.type == pygame.KEYUP:
            #     if event.key == pygame.K_SPACE:
            #         action[1] = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action[0] = 3 # Left
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action[0] = 4 # Right
        else:
            action[0] = 0 # No-op
            
        action[1] = 1 if keys[pygame.K_SPACE] else 0
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

        if done:
            print(f"Game Over! Final Info: {info}")
            # Wait a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False
            action = [0, 0, 0]

    env.close()