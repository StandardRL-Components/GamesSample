import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move, Space to shoot, Shift to reload."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive hordes of zombies in a frantic, isometric arena shooter. Manage your ammo and stay alive for 5 minutes."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 3000 # 5 minutes at 10 steps/sec

    # Colors
    COLOR_BG = (25, 25, 30)
    COLOR_ARENA = (50, 50, 60)
    COLOR_PLAYER = (50, 200, 255)
    COLOR_PLAYER_GLOW = (50, 200, 255, 50)
    COLOR_ZOMBIE = (200, 50, 50)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_MUZZLE_FLASH = (255, 220, 50)
    COLOR_BLOOD = (140, 20, 20)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_HEALTH_BG = (80, 0, 0)
    COLOR_UI_HEALTH_FG = (0, 200, 0)
    COLOR_UI_TIMER = (220, 50, 50)

    # Player
    PLAYER_SPEED = 4.0
    PLAYER_RADIUS = 12
    PLAYER_HEALTH_MAX = 100
    PLAYER_AMMO_MAX = 30
    SHOOT_COOLDOWN_MAX = 5 # steps
    RELOAD_TIME = 20 # steps

    # Zombie
    ZOMBIE_RADIUS = 11
    ZOMBIE_HEALTH_MAX = 10
    ZOMBIE_DAMAGE = 10

    # Projectile
    PROJECTILE_SPEED = 15.0
    PROJECTILE_DAMAGE = 10
    PROJECTILE_RADIUS = 3

    # Game Progression
    DIFFICULTY_INTERVAL = 600 # steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        self.player_pos = None
        self.player_health = None
        self.player_ammo = None
        self.player_aim_angle = None
        self.last_move_vec = None
        self.shoot_cooldown = None
        self.reloading = None
        self.reload_timer = None

        self.zombies = []
        self.projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.zombie_spawn_cooldown = None
        self.initial_zombie_spawn_rate = 60 # steps
        self.zombie_spawn_rate = None
        self.initial_zombie_speed = 1.0
        self.zombie_speed = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_ammo = self.PLAYER_AMMO_MAX
        self.last_move_vec = np.array([0, -1], dtype=np.float32) # Start aiming up
        self.player_aim_angle = math.atan2(-self.last_move_vec[1], self.last_move_vec[0])
        self.shoot_cooldown = 0
        self.reloading = False
        self.reload_timer = 0

        self.zombies = []
        self.projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.zombie_spawn_rate = self.initial_zombie_spawn_rate
        self.zombie_spawn_cooldown = self.zombie_spawn_rate
        self.zombie_speed = self.initial_zombie_speed
        
        for _ in range(3): # Start with a few zombies
            self._spawn_zombie()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        # --- Handle Input and Player Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_player_actions(movement, space_held, shift_held)
        reward_from_actions = self._update_player_state(space_held)
        reward += reward_from_actions

        # --- Update Game World ---
        self._update_difficulty()
        self._update_zombie_spawner()
        
        # --- Update Entities ---
        self._update_projectiles()
        self.projectiles, zombie_hits = self._check_projectile_collisions()
        reward += 0.1 * zombie_hits # Reward for hitting a zombie

        self._update_zombies()
        killed_zombies, player_was_hit = self._check_zombie_collisions()
        reward += 1.0 * killed_zombies
        if player_was_hit:
            pass # Damage is handled in _check_zombie_collisions

        self._update_particles()
        
        # self.score is for display only, reward is the step return
        cumulative_reward_before_termination = self.score + reward
        terminated = self._check_termination()

        if terminated and not self.win:
            reward -= 10 # Penalty for dying
        elif terminated and self.win:
            reward += 100 # Bonus for winning

        self.score = cumulative_reward_before_termination + (reward if not terminated else (reward - cumulative_reward_before_termination))


        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_actions(self, movement, space_held, shift_held):
        # --- Movement and Aiming ---
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_vec[1] = -1 # Up
        elif movement == 2: move_vec[1] = 1 # Down
        elif movement == 3: move_vec[0] = -1 # Left
        elif movement == 4: move_vec[0] = 1 # Right

        if np.linalg.norm(move_vec) > 0:
            self.last_move_vec = move_vec
            self.player_pos += move_vec * self.PLAYER_SPEED
            self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
            self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)
        
        self.player_aim_angle = math.atan2(-self.last_move_vec[1], self.last_move_vec[0])

        # --- Reloading ---
        if shift_held and not self.reloading and self.player_ammo < self.PLAYER_AMMO_MAX:
            self.reloading = True
            self.reload_timer = self.RELOAD_TIME
            # sfx: reload_start.wav

    def _update_player_state(self, space_held):
        reward = 0
        # --- Update Cooldowns and Timers ---
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        
        if self.reloading:
            self.reload_timer -= 1
            if self.reload_timer <= 0:
                self.reloading = False
                self.player_ammo = self.PLAYER_AMMO_MAX
                # sfx: reload_complete.wav
        
        # --- Shooting ---
        if space_held and self.player_ammo > 0 and self.shoot_cooldown == 0 and not self.reloading:
            self.player_ammo -= 1
            self.shoot_cooldown = self.SHOOT_COOLDOWN_MAX
            
            proj_pos = self.player_pos + self.last_move_vec * (self.PLAYER_RADIUS + 5)
            self.projectiles.append({
                'pos': proj_pos,
                'vel': self.last_move_vec * self.PROJECTILE_SPEED
            })
            # sfx: shoot.wav
            reward -= 0.01 # Penalty for shooting, offset by hit reward

            # Muzzle flash
            flash_pos = self.player_pos + self.last_move_vec * (self.PLAYER_RADIUS + 10)
            for _ in range(5):
                self.particles.append({
                    'pos': flash_pos.copy(),
                    'vel': self.np_random.standard_normal(2) * 2,
                    'life': self.np_random.integers(3, 6),
                    'radius': self.np_random.integers(3, 8),
                    'color': self.COLOR_MUZZLE_FLASH,
                    'type': 'spark'
                })
        return reward

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.zombie_spawn_rate = max(15, self.zombie_spawn_rate - 10)
            self.zombie_speed += 0.2
            assert self.zombie_speed > 0, "Zombie speed must be positive"

    def _update_zombie_spawner(self):
        self.zombie_spawn_cooldown -= 1
        if self.zombie_spawn_cooldown <= 0:
            self._spawn_zombie()
            self.zombie_spawn_cooldown = self.zombie_spawn_rate

    def _spawn_zombie(self):
        edge = self.np_random.integers(0, 4)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ZOMBIE_RADIUS], dtype=np.float32)
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ZOMBIE_RADIUS], dtype=np.float32)
        elif edge == 2: # Left
            pos = np.array([-self.ZOMBIE_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT)], dtype=np.float32)
        else: # Right
            pos = np.array([self.SCREEN_WIDTH + self.ZOMBIE_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT)], dtype=np.float32)
        
        self.zombies.append({
            'pos': pos,
            'health': self.ZOMBIE_HEALTH_MAX
        })
        assert self.zombies[-1]['health'] <= self.ZOMBIE_HEALTH_MAX, "Zombie health exceeds max on spawn"

    def _update_projectiles(self):
        for p in self.projectiles:
            p['pos'] += p['vel']
        
        self.projectiles = [p for p in self.projectiles if 
                            -50 < p['pos'][0] < self.SCREEN_WIDTH + 50 and 
                            -50 < p['pos'][1] < self.SCREEN_HEIGHT + 50]

    def _update_zombies(self):
        for z in self.zombies:
            direction = self.player_pos - z['pos']
            dist = np.linalg.norm(direction)
            if dist > 1:
                direction /= dist
            z['pos'] += direction * self.zombie_speed

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['type'] == 'blood':
                p['vel'] *= 0.8 # friction
        
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_projectile_collisions(self):
        live_projectiles = []
        hits = 0
        for p in self.projectiles:
            hit = False
            for z in self.zombies:
                if np.linalg.norm(p['pos'] - z['pos']) < self.ZOMBIE_RADIUS + self.PROJECTILE_RADIUS:
                    z['health'] -= self.PROJECTILE_DAMAGE
                    # sfx: hit_zombie.wav
                    hits += 1
                    hit = True
                    # Blood splatter
                    for _ in range(10):
                        angle = self.np_random.uniform(0, 2 * math.pi)
                        speed = self.np_random.uniform(1, 4)
                        vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                        self.particles.append({
                            'pos': z['pos'].copy(),
                            'vel': vel,
                            'life': self.np_random.integers(10, 20),
                            'radius': self.np_random.integers(1, 4),
                            'color': self.COLOR_BLOOD,
                            'type': 'blood'
                        })
                    break # Projectile can only hit one zombie
            if not hit:
                live_projectiles.append(p)
        return live_projectiles, hits

    def _check_zombie_collisions(self):
        surviving_zombies = []
        killed_count = 0
        player_hit = False

        for z in self.zombies:
            if z['health'] > 0:
                if np.linalg.norm(z['pos'] - self.player_pos) < self.ZOMBIE_RADIUS + self.PLAYER_RADIUS:
                    self.player_health -= self.ZOMBIE_DAMAGE
                    player_hit = True
                    # sfx: player_hit.wav
                else:
                    surviving_zombies.append(z)
            else:
                killed_count += 1
                # sfx: zombie_die.wav

        self.zombies = surviving_zombies
        assert self.player_health <= self.PLAYER_HEALTH_MAX, "Player health cannot exceed max"
        return killed_count, player_hit
    
    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            self.win = False
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_iso_rhombus(self, surface, color, center, width, height, outline_color=None, outline_width=1):
        cx, cy = int(center[0]), int(center[1])
        w, h = int(width / 2), int(height / 2)
        points = [(cx, cy - h), (cx + w, cy), (cx, cy + h), (cx - w, cy)]
        pygame.draw.polygon(surface, color, points)
        if outline_color:
            pygame.draw.polygon(surface, outline_color, points, outline_width)

    def _render_game(self):
        # Draw arena floor
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (10, 10, self.SCREEN_WIDTH-20, self.SCREEN_HEIGHT-20))
        
        # Render particles (underneath entities)
        for p in self.particles:
            if p['type'] == 'blood':
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

        # Create a list of all entities to be drawn and sort by y-coordinate for pseudo-3D
        render_list = [{'type': 'player', 'pos': self.player_pos, 'radius': self.PLAYER_RADIUS}]
        for z in self.zombies:
            render_list.append({'type': 'zombie', 'pos': z['pos'], 'radius': self.ZOMBIE_RADIUS})
        
        render_list.sort(key=lambda e: e['pos'][1] + e['radius'])

        for entity in render_list:
            if entity['type'] == 'player':
                # Player glow
                glow_surf = pygame.Surface((self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 4), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2), self.PLAYER_RADIUS * 2)
                self.screen.blit(glow_surf, (int(self.player_pos[0] - self.PLAYER_RADIUS*2), int(self.player_pos[1] - self.PLAYER_RADIUS*2)))
                # Player body
                self._draw_iso_rhombus(self.screen, self.COLOR_PLAYER, self.player_pos, self.PLAYER_RADIUS * 1.5, self.PLAYER_RADIUS * 1.5)
                # Player aim indicator
                aim_end = self.player_pos + self.last_move_vec * self.PLAYER_RADIUS
                pygame.draw.line(self.screen, (255,255,255), (int(self.player_pos[0]), int(self.player_pos[1])), (int(aim_end[0]), int(aim_end[1])), 2)

            elif entity['type'] == 'zombie':
                self._draw_iso_rhombus(self.screen, self.COLOR_ZOMBIE, entity['pos'], self.ZOMBIE_RADIUS * 1.5, self.ZOMBIE_RADIUS * 1.5)

        # Render projectiles (on top)
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

        # Render spark particles (on top)
        for p in self.particles:
            if p['type'] == 'spark':
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.player_health / self.PLAYER_HEALTH_MAX)
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_FG, (10, 10, int(200 * health_pct), 20))

        # Ammo Count
        ammo_text = self.font_small.render(f"AMMO: {self.player_ammo}/{self.PLAYER_AMMO_MAX}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (10, 35))
        if self.reloading:
            reload_text = self.font_small.render("RELOADING...", True, self.COLOR_UI_TIMER)
            self.screen.blit(reload_text, (10, 55))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 10.0 # Assuming 10 steps/sec
        time_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TIMER)
        text_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_text, text_rect)
        
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 35))
        self.screen.blit(score_text, score_rect)

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                end_text = self.font_large.render("YOU SURVIVED", True, self.COLOR_PLAYER)
            else:
                end_text = self.font_large.render("YOU DIED", True, self.COLOR_UI_TIMER)
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "zombies": len(self.zombies),
            "win": self.win,
        }
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # You must unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    total_reward = 0
    
    # Pygame setup for human play
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()

    while running:
        # Action defaults
        movement = 0 # none
        space_held = 0 # released
        shift_held = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Survived: {info['win']}")
            # Render final frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            # Wait a bit before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0
        else:
            # Render the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        # Control game speed to ~10 steps per second for human play
        clock.tick(10)
        
    env.close()