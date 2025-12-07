
# Generated: 2025-08-27T20:45:23.434952
# Source Brief: brief_02566.md
# Brief Index: 2566

        
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
        "Controls: Arrow keys to move. Hold Space to shoot. "
        "Shift is not used in this version."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a relentless zombie horde in a top-down arena shooter. "
        "Survive for 5 minutes to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Game world
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30
    WALL_THICKNESS = 10
    MAX_EPISODE_STEPS = 300 * TARGET_FPS # 5 minutes at 30 FPS

    # Player
    PLAYER_SIZE = 20
    PLAYER_SPEED = 4
    PLAYER_START_HEALTH = 100
    PLAYER_START_AMMO = 100
    PLAYER_FIRE_COOLDOWN = 5 # frames
    PLAYER_IFRAMES = 15 # invincibility frames after getting hit

    # Zombies
    ZOMBIE_SIZE = 18
    ZOMBIE_START_SPEED = 0.8
    ZOMBIE_HEALTH = 10
    ZOMBIE_DAMAGE = 10
    ZOMBIE_START_COUNT = 5
    ZOMBIE_SPAWN_RATE = 150 # steps
    ZOMBIE_SPEED_INCREASE_RATE = 600 # steps
    ZOMBIE_SPEED_INCREASE_AMOUNT = 0.1
    ZOMBIE_MAX_COUNT = 100

    # Projectiles
    PROJECTILE_SPEED = 12
    PROJECTILE_DAMAGE = 10
    PROJECTILE_WIDTH = 10
    PROJECTILE_HEIGHT = 4

    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_WALL = (80, 80, 90)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_IFRAME = (150, 255, 200)
    COLOR_ZOMBIE = (255, 80, 80)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_BAR_BG = (100, 40, 40)
    COLOR_HEALTH_BAR_FG = (100, 220, 100)
    COLOR_AMMO_BAR_BG = (80, 80, 80)
    COLOR_AMMO_BAR_FG = (100, 150, 255)

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
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_timer = pygame.font.SysFont("monospace", 24, bold=True)
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.player_health = None
        self.player_ammo = None
        self.player_fire_cooldown = None
        self.player_iframe_timer = None
        self.prev_space_held = None
        
        self.zombies = None
        self.projectiles = None
        
        self.zombie_current_speed = None
        self.zombie_spawn_timer = None
        
        self.steps = None
        self.score = None
        self.game_over = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        self.player_health = self.PLAYER_START_HEALTH
        self.player_ammo = self.PLAYER_START_AMMO
        self.player_fire_cooldown = 0
        self.player_iframe_timer = 0
        self.prev_space_held = False

        # Game state
        self.zombies = []
        self.projectiles = []
        self.zombie_current_speed = self.ZOMBIE_START_SPEED
        self.zombie_spawn_timer = 0
        
        for _ in range(self.ZOMBIE_START_COUNT):
            self._spawn_zombie(is_initial_spawn=True)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(self.TARGET_FPS)
        step_reward = 0.1 # Survival reward

        # --- Unpack Action ---
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is unused per brief

        # --- Update Game Logic ---
        self._handle_player_movement(movement)
        self._handle_player_shooting(space_held)
        
        step_reward += self._update_projectiles()
        step_reward += self._update_zombies()
        
        self._update_spawning()
        self._update_difficulty()

        # Update timers
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
        if self.player_iframe_timer > 0:
            self.player_iframe_timer -= 1
        
        self.prev_space_held = space_held
        self.steps += 1
        self.score += step_reward

        # --- Check Termination ---
        terminated = False
        if self.player_health <= 0:
            step_reward = -100.0 # Death penalty
            self.score += step_reward
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_EPISODE_STEPS:
            step_reward = 100.0 # Survival bonus
            self.score += step_reward
            terminated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        if movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.WALL_THICKNESS, self.SCREEN_WIDTH - self.WALL_THICKNESS - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.WALL_THICKNESS, self.SCREEN_HEIGHT - self.WALL_THICKNESS - self.PLAYER_SIZE)

    def _handle_player_shooting(self, space_held):
        # Shoot on key press (rising edge)
        is_shooting = space_held and not self.prev_space_held
        if is_shooting and self.player_fire_cooldown <= 0 and self.player_ammo > 0:
            # Find nearest zombie to target
            nearest_zombie = None
            min_dist_sq = float('inf')
            
            for zombie in self.zombies:
                dist_sq = (zombie['pos'][0] - self.player_pos[0])**2 + (zombie['pos'][1] - self.player_pos[1])**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    nearest_zombie = zombie
            
            if nearest_zombie:
                # # sound: player_shoot.wav
                self.player_ammo -= 1
                self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN
                
                dx = nearest_zombie['pos'][0] - self.player_pos[0]
                dy = nearest_zombie['pos'][1] - self.player_pos[1]
                dist = math.hypot(dx, dy)
                
                if dist > 0:
                    vel = [dx / dist * self.PROJECTILE_SPEED, dy / dist * self.PROJECTILE_SPEED]
                    start_pos = [self.player_pos[0] + self.PLAYER_SIZE/2, self.player_pos[1] + self.PLAYER_SIZE/2]
                    self.projectiles.append({'pos': start_pos, 'vel': vel})

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            
            # Check boundaries
            if not (0 < p['pos'][0] < self.SCREEN_WIDTH and 0 < p['pos'][1] < self.SCREEN_HEIGHT):
                continue

            hit = False
            proj_rect = pygame.Rect(p['pos'][0], p['pos'][1], self.PROJECTILE_WIDTH, self.PROJECTILE_HEIGHT)
            
            for z in self.zombies:
                zombie_rect = pygame.Rect(z['pos'][0], z['pos'][1], self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
                if proj_rect.colliderect(zombie_rect):
                    # # sound: zombie_hit.wav
                    z['health'] -= self.PROJECTILE_DAMAGE
                    reward += 1 # Hit reward
                    hit = True
                    break # Projectile can only hit one zombie
            
            if not hit:
                projectiles_to_keep.append(p)

        self.projectiles = projectiles_to_keep
        return reward

    def _update_zombies(self):
        reward = 0
        zombies_alive = []
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        for z in self.zombies:
            if z['health'] <= 0:
                # # sound: zombie_die.wav
                reward += 2 # Kill reward
                continue
            
            # Move zombie towards player
            dx = self.player_pos[0] - z['pos'][0]
            dy = self.player_pos[1] - z['pos'][1]
            dist = math.hypot(dx, dy)
            if dist > 0:
                z['pos'][0] += dx / dist * self.zombie_current_speed
                z['pos'][1] += dy / dist * self.zombie_current_speed

            # Check collision with player
            zombie_rect = pygame.Rect(z['pos'][0], z['pos'][1], self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            if player_rect.colliderect(zombie_rect) and self.player_iframe_timer <= 0:
                # # sound: player_hurt.wav
                self.player_health -= self.ZOMBIE_DAMAGE
                self.player_iframe_timer = self.PLAYER_IFRAMES
                # Small knockback
                if dist > 0:
                   self.player_pos[0] += -dx / dist * 15
                   self.player_pos[1] += -dy / dist * 15
                   self._handle_player_movement(0) # Re-check boundaries

            zombies_alive.append(z)
        
        self.zombies = zombies_alive
        return reward

    def _spawn_zombie(self, is_initial_spawn=False):
        if len(self.zombies) >= self.ZOMBIE_MAX_COUNT:
            return

        # Spawn away from the player
        while True:
            x = self.np_random.integers(self.WALL_THICKNESS, self.SCREEN_WIDTH - self.WALL_THICKNESS - self.ZOMBIE_SIZE)
            y = self.np_random.integers(self.WALL_THICKNESS, self.SCREEN_HEIGHT - self.WALL_THICKNESS - self.ZOMBIE_SIZE)
            dist_to_player = math.hypot(x - self.player_pos[0], y - self.player_pos[1])
            if not is_initial_spawn and dist_to_player < 150:
                continue # Too close, try again
            if is_initial_spawn and dist_to_player < 50:
                continue
            break
            
        self.zombies.append({'pos': [x, y], 'health': self.ZOMBIE_HEALTH})

    def _update_spawning(self):
        self.zombie_spawn_timer += 1
        if self.zombie_spawn_timer >= self.ZOMBIE_SPAWN_RATE:
            self.zombie_spawn_timer = 0
            self._spawn_zombie()

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.ZOMBIE_SPEED_INCREASE_RATE == 0:
            self.zombie_current_speed += self.ZOMBIE_SPEED_INCREASE_AMOUNT

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.SCREEN_HEIGHT - self.WALL_THICKNESS, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))

        # Draw zombies
        for z in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, (int(z['pos'][0]), int(z['pos'][1]), self.ZOMBIE_SIZE, self.ZOMBIE_SIZE))

        # Draw projectiles
        for p in self.projectiles:
            start_pos = (int(p['pos'][0]), int(p['pos'][1] + self.PROJECTILE_HEIGHT / 2))
            end_pos = (int(p['pos'][0] - p['vel'][0]*0.5), int(p['pos'][1] - p['vel'][1]*0.5 + self.PROJECTILE_HEIGHT / 2))
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 3)

        # Draw player
        player_color = self.COLOR_PLAYER if self.player_iframe_timer % 4 < 2 else self.COLOR_PLAYER_IFRAME
        player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, player_color, player_rect)
        pygame.draw.rect(self.screen, (255,255,255), player_rect, 1) # Outline

    def _render_ui(self):
        # Health bar
        health_ratio = max(0, self.player_health / self.PLAYER_START_HEALTH)
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (15, 15, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (15, 15, int(bar_width * health_ratio), bar_height))
        health_text = self.font_main.render(f"HP", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15 + bar_width + 5, 15))

        # Ammo count
        ammo_text = self.font_main.render(f"AMMO: {self.player_ammo}", True, self.COLOR_TEXT)
        text_rect = ammo_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 15))
        self.screen.blit(ammo_text, text_rect)
        
        # Timer
        time_left = max(0, (self.MAX_EPISODE_STEPS - self.steps) / self.TARGET_FPS)
        minutes = int(time_left // 60)
        seconds = int(time_left % 60)
        timer_text = self.font_timer.render(f"{minutes:02}:{seconds:02}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(midtop=(self.SCREEN_WIDTH / 2, 15))
        self.screen.blit(timer_text, timer_rect)

        # Game Over text
        if self.game_over:
            outcome_text = "VICTORY!" if self.player_health > 0 else "GAME OVER"
            go_text_surf = self.font_timer.render(outcome_text, True, self.COLOR_TEXT)
            go_text_rect = go_text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            pygame.draw.rect(self.screen, (0,0,0,150), go_text_rect.inflate(20, 20))
            self.screen.blit(go_text_surf, go_text_rect)

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

    def validate_implementation(self):
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
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()
    
    terminated = False
    while not terminated:
        # --- Player Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        clock.tick(GameEnv.TARGET_FPS)

    env.close()
    print(f"Game Over. Final Score: {info['score']:.2f}, Steps: {info['steps']}")