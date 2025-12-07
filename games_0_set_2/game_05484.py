import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a top-down zombie survival game.
    The player must survive an endless horde of zombies for a fixed amount of time.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to shoot. Hold Shift near ammo to reload."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive the relentless zombie horde! Scavenge for ammo, prioritize your targets, "
        "and hold out as long as you can in this top-down arcade shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1000
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_ARENA = (30, 35, 55)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_ZOMBIE = (255, 80, 80)
    COLOR_AMMO = (255, 220, 50)
    COLOR_BULLET = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR = (50, 200, 50)
    COLOR_HEALTH_BAR_BG = (100, 40, 40)
    COLOR_SHADOW = (0, 0, 0, 50)

    # Player
    PLAYER_SIZE = 12
    PLAYER_SPEED = 3.5
    MAX_HEALTH = 10
    MAX_AMMO = 30
    
    # Zombies
    ZOMBIE_SIZE = 12
    ZOMBIE_SPEED = 1.2
    ZOMBIE_HEALTH = 1
    INITIAL_ZOMBIE_SPAWN_PERIOD = 100
    DIFFICULTY_SCALING = 0.05 # Period shortens by this amount each step
    
    # Weapons & Pickups
    BULLET_SPEED = 10
    BULLET_COOLDOWN_STEPS = 4
    AMMO_DROP_ON_KILLS = 5
    RELOAD_RADIUS = 20
    AMMO_DROP_SIZE = 10

    # Rewards
    REWARD_STEP = 0.01
    REWARD_KILL = 1.0
    REWARD_WIN = 50.0
    REWARD_LOSE = -100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # This attribute is only for seeding the RNG
        self.np_random = None

        # Initialize state variables to None
        self.steps = 0
        self.score = 0
        self.player_rect = None
        self.player_health = 0
        self.player_ammo = 0
        self.last_move_vector = None
        self.shot_cooldown = 0
        
        self.zombies = []
        self.ammo_drops = []
        self.bullets = []
        self.particles = []
        
        self.zombie_spawn_timer = 0
        self.zombie_spawn_period = self.INITIAL_ZOMBIE_SPAWN_PERIOD
        self.zombies_killed_since_drop = 0
        
        self.game_over = False
        
        # This will call reset() and initialize the state
        # self.validate_implementation() # Commented out for submission, can be used for local testing
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_rect = pygame.Rect(self.WIDTH / 2 - self.PLAYER_SIZE / 2, 
                                       self.HEIGHT / 2 - self.PLAYER_SIZE / 2, 
                                       self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_health = self.MAX_HEALTH
        self.player_ammo = self.MAX_AMMO
        self.last_move_vector = pygame.Vector2(0, -1) # Start aiming up
        self.shot_cooldown = 0
        
        self.zombies = []
        self.ammo_drops = []
        self.bullets = []
        self.particles = []
        
        self.zombie_spawn_timer = 0
        self.zombie_spawn_period = self.INITIAL_ZOMBIE_SPAWN_PERIOD
        self.zombies_killed_since_drop = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = self.REWARD_STEP
        self.steps += 1
        
        reward += self._handle_input(action)
        reward += self._update_game_state()
        self._handle_spawning()
        
        terminated = self._check_termination()
        truncated = False # No truncation condition in this game
        if terminated:
            if self.player_health <= 0:
                reward += self.REWARD_LOSE
            elif self.steps >= self.MAX_STEPS:
                reward += self.REWARD_WIN
            self.game_over = True
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        move_vector = pygame.Vector2(0, 0)
        if movement == 1: move_vector.y = -1 # Up
        elif movement == 2: move_vector.y = 1  # Down
        elif movement == 3: move_vector.x = -1 # Left
        elif movement == 4: move_vector.x = 1  # Right
        
        if move_vector.length() > 0:
            move_vector.normalize_ip()
            self.last_move_vector = pygame.math.Vector2(move_vector) # FIX: Vector2 does not have .copy()
            self.player_rect.move_ip(move_vector * self.PLAYER_SPEED)
        
        self.player_rect.clamp_ip(self.screen.get_rect())

        # Shooting
        if self.shot_cooldown > 0:
            self.shot_cooldown -= 1
            
        if space_held and self.player_ammo > 0 and self.shot_cooldown <= 0:
            self.player_ammo -= 1
            self.shot_cooldown = self.BULLET_COOLDOWN_STEPS
            
            bullet_start_pos = self.player_rect.center + self.last_move_vector * (self.PLAYER_SIZE / 2)
            bullet_rect = pygame.Rect(bullet_start_pos.x - 2, bullet_start_pos.y - 2, 4, 4)
            self.bullets.append({'rect': bullet_rect, 'vel': self.last_move_vector * self.BULLET_SPEED})
            
            # Muzzle flash
            for _ in range(5):
                vel = self.last_move_vector * self.np_random.uniform(1, 3) + pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2))
                self.particles.append({'pos': list(bullet_start_pos), 'vel': list(vel), 'life': 5, 'size': self.np_random.integers(1, 4), 'color': self.COLOR_AMMO})

        # Reloading
        if shift_held:
            for drop in self.ammo_drops[:]:
                if pygame.Vector2(self.player_rect.center).distance_to(drop.center) < self.RELOAD_RADIUS:
                    self.player_ammo = self.MAX_AMMO
                    self.ammo_drops.remove(drop)
                    # Pickup effect
                    for i in range(20):
                        angle = i * (360 / 20)
                        vel = pygame.Vector2(1, 0).rotate(angle) * self.np_random.uniform(1, 3)
                        self.particles.append({'pos': list(drop.center), 'vel': list(vel), 'life': 10, 'size': self.np_random.integers(2, 4), 'color': self.COLOR_AMMO})
                    break # Reload from one drop at a time
        
        return 0 # Input handling itself doesn't generate reward

    def _update_game_state(self):
        reward = 0
        
        # Update bullets
        for bullet in self.bullets[:]:
            bullet['rect'].move_ip(bullet['vel'])
            if not self.screen.get_rect().colliderect(bullet['rect']):
                self.bullets.remove(bullet)

        # Update zombies
        for zombie in self.zombies:
            direction = pygame.Vector2(self.player_rect.center) - pygame.Vector2(zombie.center)
            if direction.length() > 0:
                direction.normalize_ip()
            zombie.move_ip(direction * self.ZOMBIE_SPEED)
            
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Collisions: Bullets vs Zombies
        for bullet in self.bullets[:]:
            collided = False
            for zombie in self.zombies[:]:
                if bullet['rect'].colliderect(zombie):
                    self.bullets.remove(bullet)
                    self.zombies.remove(zombie)
                    reward += self.REWARD_KILL
                    self.score += 1
                    self.zombies_killed_since_drop += 1
                    # Death particle effect
                    for _ in range(30):
                        vel = pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3))
                        self.particles.append({'pos': list(zombie.center), 'vel': list(vel), 'life': self.np_random.integers(10, 20), 'size': self.np_random.integers(1, 4), 'color': self.COLOR_ZOMBIE})
                    collided = True
                    break
            if collided:
                continue


        # Collisions: Player vs Zombies
        for zombie in self.zombies[:]:
            if self.player_rect.colliderect(zombie):
                self.player_health -= self.ZOMBIE_HEALTH
                self.zombies.remove(zombie)
                # Player damage effect
                for _ in range(15):
                    vel = pygame.Vector2(self.np_random.uniform(-4, 4), self.np_random.uniform(-4, 4))
                    self.particles.append({'pos': list(self.player_rect.center), 'vel': list(vel), 'life': 8, 'size': self.np_random.integers(1, 3), 'color': self.COLOR_PLAYER})

        return reward

    def _handle_spawning(self):
        # Zombie spawning
        self.zombie_spawn_timer += 1
        self.zombie_spawn_period = max(20, self.INITIAL_ZOMBIE_SPAWN_PERIOD - self.steps * self.DIFFICULTY_SCALING)
        
        if self.zombie_spawn_timer >= self.zombie_spawn_period:
            self.zombie_spawn_timer = 0
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = (self.np_random.integers(self.WIDTH), -self.ZOMBIE_SIZE)
            elif edge == 1: # Bottom
                pos = (self.np_random.integers(self.WIDTH), self.HEIGHT)
            elif edge == 2: # Left
                pos = (-self.ZOMBIE_SIZE, self.np_random.integers(self.HEIGHT))
            else: # Right
                pos = (self.WIDTH, self.np_random.integers(self.HEIGHT))
            
            self.zombies.append(pygame.Rect(pos[0], pos[1], self.ZOMBIE_SIZE, self.ZOMBIE_SIZE))

        # Ammo drop spawning
        if self.zombies_killed_since_drop >= self.AMMO_DROP_ON_KILLS:
            self.zombies_killed_since_drop = 0
            drop_pos = (self.np_random.integers(50, self.WIDTH - 50),
                        self.np_random.integers(50, self.HEIGHT - 50))
            self.ammo_drops.append(pygame.Rect(drop_pos[0], drop_pos[1], self.AMMO_DROP_SIZE, self.AMMO_DROP_SIZE))
            
    def _check_termination(self):
        return self.player_health <= 0 or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw arena
        pygame.draw.rect(self.screen, self.COLOR_ARENA, self.screen.get_rect(), width=20)
        
        # Draw shadows
        self._draw_shadow(self.player_rect)
        for z in self.zombies: self._draw_shadow(z)
        for a in self.ammo_drops: self._draw_shadow(a)
        
        # Draw ammo drops
        for drop in self.ammo_drops:
            pygame.draw.rect(self.screen, self.COLOR_AMMO, drop)
            
        # Draw zombies
        for zombie in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, zombie)

        # Draw player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)
        
        # Draw aiming direction indicator
        aim_end = pygame.Vector2(self.player_rect.center) + self.last_move_vector * (self.PLAYER_SIZE)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, self.player_rect.center, aim_end, 1)

        # Draw bullets
        for bullet in self.bullets:
            pygame.draw.rect(self.screen, self.COLOR_BULLET, bullet['rect'])

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), p['color'])

    def _draw_shadow(self, rect):
        shadow_rect = rect.copy()
        shadow_rect.x += 3
        shadow_rect.y += 3
        shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        shadow_surf.fill(self.COLOR_SHADOW)
        self.screen.blit(shadow_surf, shadow_rect.topleft)

    def _render_ui(self):
        # Health bar
        health_ratio = max(0, self.player_health / self.MAX_HEALTH)
        bar_width = 150
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, bar_height))

        # Ammo count
        ammo_text = self.font_small.render(f"AMMO: {self.player_ammo}/{self.MAX_AMMO}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (self.WIDTH - ammo_text.get_width() - 10, 10))
        
        # Timer
        time_left = self.MAX_STEPS - self.steps
        time_text = self.font_large.render(f"{time_left//self.metadata['render_fps']}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH / 2 - time_text.get_width() / 2, 5))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 35))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            result_text_str = "YOU SURVIVED" if self.player_health > 0 else "GAME OVER"
            result_text = self.font_large.render(result_text_str, True, self.COLOR_PLAYER if self.player_health > 0 else self.COLOR_ZOMBIE)
            self.screen.blit(result_text, (self.WIDTH / 2 - result_text.get_width() / 2, self.HEIGHT / 2 - result_text.get_height() / 2 - 20))
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            self.screen.blit(final_score_text, (self.WIDTH / 2 - final_score_text.get_width() / 2, self.HEIGHT / 2 + 20))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "zombies": len(self.zombies),
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    import random
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # --- Pygame setup for human viewing ---
    pygame.display.quit() # Quit the dummy display
    pygame.init() # Re-init with default video driver
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Zombie Survival - Demo")
    clock = pygame.time.Clock()
    running = True
    terminated = False

    while running:
        # Get action from keyboard
        action = [0, 0, 0] # Default: no move, no shoot, no reload
        if not terminated:
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event and reset on termination
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r and terminated:
                    obs, info = env.reset(seed=random.randint(0, 10000))
                    terminated = False

        clock.tick(env.metadata["render_fps"])

    env.close()