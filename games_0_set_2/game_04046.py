import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame



# --- Helper Classes for Game Objects ---

class Player:
    def __init__(self, screen_width, screen_height, np_random):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.np_random = np_random
        self.size = 20
        self.color = (50, 255, 50)  # Bright Green
        self.speed = 4
        self.reset()

    def reset(self):
        self.pos = pygame.math.Vector2(self.screen_width / 2, self.screen_height / 2)
        self.health = 100
        self.max_health = 100
        self.ammo = 50
        self.max_ammo = 100
        self.last_move_dir = pygame.math.Vector2(0, -1)  # Default aim up

    def update(self, movement_action):
        move_vec = pygame.math.Vector2(0, 0)
        if movement_action == 1:  # Up
            move_vec.y = -1
        elif movement_action == 2:  # Down
            move_vec.y = 1
        elif movement_action == 3:  # Left
            move_vec.x = -1
        elif movement_action == 4:  # Right
            move_vec.x = 1

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.pos += move_vec * self.speed
            self.last_move_dir = pygame.math.Vector2(move_vec)

        # Clamp position to stay within arena bounds
        arena_margin = 20
        self.pos.x = np.clip(self.pos.x, arena_margin + self.size / 2, self.screen_width - arena_margin - self.size / 2)
        self.pos.y = np.clip(self.pos.y, arena_margin + self.size / 2, self.screen_height - arena_margin - self.size / 2)

    def take_damage(self, amount):
        self.health = max(0, self.health - amount)

    def add_ammo(self, amount):
        self.ammo = min(self.max_ammo, self.ammo + amount)

    def get_rect(self):
        return pygame.Rect(self.pos.x - self.size / 2, self.pos.y - self.size / 2, self.size, self.size)

    def draw(self, surface):
        rect = self.get_rect()
        pygame.draw.rect(surface, self.color, rect)
        pygame.draw.rect(surface, tuple(c/1.5 for c in self.color), rect, 2) # Border

class Zombie:
    def __init__(self, pos, speed):
        self.pos = pygame.math.Vector2(pos)
        self.size = 18
        self.color = (255, 50, 50)  # Bright Red
        self.speed = speed
        self.health = 1

    def update(self, player_pos):
        direction = player_pos - self.pos
        if direction.length() > 0:
            direction.normalize_ip()
        self.pos += direction * self.speed

    def get_rect(self):
        return pygame.Rect(self.pos.x - self.size / 2, self.pos.y - self.size / 2, self.size, self.size)

    def draw(self, surface):
        rect = self.get_rect()
        pygame.draw.rect(surface, self.color, rect)

class Bullet:
    def __init__(self, pos, direction):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(direction).normalize() * 12
        self.size = 5
        self.color = (255, 255, 200) # Pale Yellow

    def update(self):
        self.pos += self.vel

    def get_rect(self):
        return pygame.Rect(self.pos.x - self.size / 2, self.pos.y - self.size / 2, self.size, self.size)

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), self.size)

class AmmoPack:
    def __init__(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.size = 15
        self.color = (255, 220, 0) # Bright Yellow
        self.pulse_timer = 0
        self.pulse_speed = 0.1

    def update(self):
        self.pulse_timer += self.pulse_speed

    def get_rect(self):
        return pygame.Rect(self.pos.x - self.size / 2, self.pos.y - self.size / 2, self.size, self.size)

    def draw(self, surface):
        pulse_alpha = 128 + 127 * math.sin(self.pulse_timer)
        glow_color = (*self.color, pulse_alpha)
        glow_size = self.size * (1.2 + 0.2 * math.sin(self.pulse_timer))
        
        # Draw glow
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color, (glow_size, glow_size), glow_size)
        surface.blit(glow_surf, (self.pos.x - glow_size, self.pos.y - glow_size), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Draw pack
        pygame.draw.rect(surface, self.color, self.get_rect())
        pygame.draw.rect(surface, (0,0,0), self.get_rect(), 2)

class Particle:
    def __init__(self, pos, color, lifetime, np_random):
        self.pos = pygame.math.Vector2(pos)
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = np_random.uniform(2, 6)

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95 # friction
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = 255 * (self.lifetime / self.max_lifetime)
            color_with_alpha = (*self.color, alpha)
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (self.size, self.size), self.size)
            surface.blit(temp_surf, (self.pos.x - self.size, self.pos.y - self.size), special_flags=pygame.BLEND_RGBA_ADD)

# --- Main Game Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Press Space to shoot in your last direction of movement."
    )

    game_description = (
        "Survive for as long as you can against endless hordes of zombies in a top-down arena. "
        "Collect ammo packs to keep your weapon firing. The difficulty increases over time."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # For headless execution
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 5000 # Game ends in victory if this is reached
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_timer = pygame.font.SysFont("monospace", 24, bold=True)

        # Colors
        self.COLOR_BG = (15, 15, 20)
        self.COLOR_ARENA = (100, 100, 120)
        self.COLOR_TEXT = (220, 220, 220)
        
        self.player = None
        self.zombies = []
        self.bullets = []
        self.ammo_packs = []
        self.particles = []
        self.muzzle_flash_timer = 0
        
        # This will be called once to initialize np_random
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.player is None:
            self.player = Player(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.np_random)
        self.player.reset()
        
        self.zombies = []
        self.bullets = []
        self.ammo_packs = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.muzzle_flash_timer = 0
        
        # Game progression state
        self.zombie_spawn_timer = 0
        self.ammo_spawn_timer = 0
        self.zombie_base_speed = 1.0
        self.zombie_spawn_interval = 50

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward += 0.001 # Small survival reward per step, adjusted from brief for 5000 steps

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Actions ---
        self.player.update(movement)
        
        space_pressed = space_held and not self.last_space_held
        if space_pressed and self.player.ammo > 0:
            self.player.ammo -= 1
            bullet = Bullet(self.player.pos, self.player.last_move_dir)
            self.bullets.append(bullet)
            self.muzzle_flash_timer = 3
            # # Sound: Play shoot.wav
            # Create muzzle particles
            for _ in range(5):
                self.particles.append(Particle(self.player.pos + self.player.last_move_dir * 15, (255, 255, 200), 5, self.np_random))

        self.last_space_held = space_held

        # --- Update Game State ---
        self._update_difficulty()
        self._update_spawns()
        
        for z in self.zombies:
            z.update(self.player.pos)
        for b in self.bullets:
            b.update()
        for a in self.ammo_packs:
            a.update()
        for p in self.particles:
            p.update()

        # --- Handle Collisions & Interactions ---
        # Player vs Zombies
        player_rect = self.player.get_rect()
        for z in self.zombies:
            if player_rect.colliderect(z.get_rect()):
                self.player.take_damage(1)
                # # Sound: Play player_hurt.wav

        # Bullets vs Zombies
        for b in self.bullets[:]:
            bullet_rect = b.get_rect()
            for z in self.zombies[:]:
                if bullet_rect.colliderect(z.get_rect()):
                    self.bullets.remove(b)
                    self.zombies.remove(z)
                    reward += 1.0
                    self.score += 10
                    # # Sound: Play zombie_die.wav
                    # Create death particles
                    for _ in range(20):
                        self.particles.append(Particle(z.pos, z.color, self.np_random.integers(10, 30), self.np_random))
                    break # Bullet can only hit one zombie
        
        # Player vs Ammo Packs
        for a in self.ammo_packs[:]:
            if player_rect.colliderect(a.get_rect()):
                self.ammo_packs.remove(a)
                self.player.add_ammo(25)
                reward += 0.5
                self.score += 5
                # # Sound: Play ammo_pickup.wav
                # Create pickup particles
                for _ in range(15):
                    self.particles.append(Particle(a.pos, a.color, self.np_random.integers(10, 25), self.np_random))

        # --- Cleanup ---
        self.bullets = [b for b in self.bullets if self.screen.get_rect().collidepoint(b.pos)]
        self.particles = [p for p in self.particles if p.lifetime > 0]
        if self.muzzle_flash_timer > 0:
            self.muzzle_flash_timer -= 1

        # --- Check Termination ---
        terminated = False
        truncated = False # Gymnasium standard is to use truncated for time limits
        if self.player.health <= 0:
            terminated = True
            self.game_over = True
            reward = -100.0
            # # Sound: Play game_over.wav
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            self.game_over = True
            reward = 100.0
            # # Sound: Play victory.wav
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _update_difficulty(self):
        # Per brief: spawn rate decreases by 10 steps every 600 steps, capped at 20.
        if self.steps > 0 and self.steps % 600 == 0:
            self.zombie_spawn_interval = max(20, self.zombie_spawn_interval - 10)
        
        # Per brief: speed increases by 0.1 every 1000 steps.
        if self.steps > 0 and self.steps % 1000 == 0:
            self.zombie_base_speed += 0.1

    def _update_spawns(self):
        # Spawn Zombies
        self.zombie_spawn_timer += 1
        if self.zombie_spawn_timer >= self.zombie_spawn_interval:
            self.zombie_spawn_timer = 0
            spawn_side = self.np_random.integers(4)
            if spawn_side == 0: # Top
                pos = (self.np_random.uniform(0, self.SCREEN_WIDTH), -20)
            elif spawn_side == 1: # Bottom
                pos = (self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20)
            elif spawn_side == 2: # Left
                pos = (-20, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            else: # Right
                pos = (self.SCREEN_WIDTH + 20, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            
            self.zombies.append(Zombie(pos, self.zombie_base_speed))

        # Spawn Ammo
        self.ammo_spawn_timer += 1
        if self.ammo_spawn_timer >= 100: # Fixed spawn rate from brief
            self.ammo_spawn_timer = 0
            if len(self.ammo_packs) < 3: # Limit number of packs on screen
                margin = 40
                pos = (
                    self.np_random.uniform(margin, self.SCREEN_WIDTH - margin),
                    self.np_random.uniform(margin, self.SCREEN_HEIGHT - margin)
                )
                self.ammo_packs.append(AmmoPack(pos))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Draw arena boundary
        arena_margin = 20
        arena_rect = pygame.Rect(arena_margin, arena_margin, self.SCREEN_WIDTH - 2 * arena_margin, self.SCREEN_HEIGHT - 2 * arena_margin)
        pygame.draw.rect(self.screen, self.COLOR_ARENA, arena_rect, 1)

        # Render game elements
        for p in self.particles: p.draw(self.screen)
        for a in self.ammo_packs: a.draw(self.screen)
        for z in self.zombies: z.draw(self.screen)
        for b in self.bullets: b.draw(self.screen)
        
        self.player.draw(self.screen)

        # Muzzle flash
        if self.muzzle_flash_timer > 0:
            flash_pos = self.player.pos + self.player.last_move_dir * 15
            radius = 15 * (self.muzzle_flash_timer / 3.0)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(flash_pos.x), int(flash_pos.y)), int(radius))

        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Health Bar
        health_pct = self.player.health / self.player.max_health
        health_bar_width = 200
        health_bar_height = 20
        health_bar_fill = health_bar_width * health_pct
        
        pygame.draw.rect(self.screen, (80, 0, 0), (10, 10, health_bar_width, health_bar_height))
        if health_bar_fill > 0:
            pygame.draw.rect(self.screen, (0, 180, 0), (10, 10, health_bar_fill, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, 10, health_bar_width, health_bar_height), 1)

        # Ammo Count
        ammo_text = self.font_ui.render(f"AMMO: {self.player.ammo}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (10, 35))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_timer.render(f"{time_left}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(midtop=(self.SCREEN_WIDTH / 2, 5))
        self.screen.blit(time_text, time_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player.health,
            "ammo": self.player.ammo,
            "zombies": len(self.zombies),
        }

    def close(self):
        pygame.quit()


# --- Example Usage ---
if __name__ == '__main__':
    # Set the video driver to a visible one if you want to see the game
    # Use 'x11', 'dummy', or 'windows' depending on your system
    # 'dummy' is for headless execution
    try:
        os.environ['SDL_VIDEODRIVER'] = 'x11'
        import pygame
        pygame.display.init()
    except pygame.error:
        print("No available video device. Running in headless mode.")
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    try:
        pygame.display.set_caption("Zombie Survival")
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        interactive = True
    except pygame.error:
        print("Could not create display. Manual play is disabled.")
        interactive = False

    if interactive:
        obs, info = env.reset()
        terminated = False
        
        # Map keyboard keys to MultiDiscrete actions
        key_map = {
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
        }
        
        running = True
        while running:
            # Construct action from keyboard state
            movement_action = 0 # No-op
            space_action = 0
            shift_action = 0
            
            keys = pygame.key.get_pressed()
            for key, move_val in key_map.items():
                if keys[key]:
                    movement_action = move_val
                    break # Prioritize first key found
            
            if keys[pygame.K_SPACE]:
                space_action = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_action = 1

            action = [movement_action, space_action, shift_action]
            
            # Environment step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render to screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
                # Wait a moment before resetting
                pygame.time.wait(2000)
                obs, info = env.reset()

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            env.clock.tick(60) # Control the frame rate

    env.close()