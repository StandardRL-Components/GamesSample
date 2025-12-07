import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:16:58.935128
# Source Brief: brief_00380.md
# Brief Index: 380
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A cyberpunk-themed 2.5D shooter/platformer Gymnasium environment.
    The agent navigates a vertically scrolling data stream, jumping on platforms,
    shooting digital enemies, and deploying clones to assist.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a vertically scrolling data stream in this cyberpunk shooter. "
        "Jump between platforms, shoot digital enemies, and deploy clones to assist you."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to shoot and shift to deploy a clone."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000

    # Colors
    COLOR_BG_TOP = (10, 0, 20)
    COLOR_BG_BOTTOM = (30, 0, 50)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 100, 155)
    COLOR_CLONE = (0, 150, 200)
    COLOR_PLATFORM = (0, 255, 100)
    COLOR_PLATFORM_GLOW = (0, 100, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (150, 0, 0)
    COLOR_FRAGMENT = (255, 255, 0)
    COLOR_PLAYER_BULLET = (100, 255, 255)
    COLOR_ENEMY_BULLET = (255, 100, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_PARTICLE_EXHAUST = (255, 150, 0)
    COLOR_PARTICLE_EXPLOSION = (255, 100, 0)

    # Physics & Gameplay
    PLAYER_SPEED = 4.0
    PLAYER_JUMP_POWER = -9.0
    GRAVITY = 0.4
    PLAYER_SIZE = (20, 30)
    CLONE_SIZE = (20, 30)
    ENEMY_SIZE = (25, 25)
    PLAYER_SHOOT_COOLDOWN = 8  # Steps between shots
    ENEMY_SHOOT_COOLDOWN = 60
    MAX_CLONES = 3

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_weapon = pygame.font.Font(None, 22)

        # --- Game State Initialization ---
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.player_shoot_cooldown = 0
        self.last_shift_state = 0
        self.high_score = 0

        # Lists for dynamic objects
        self.platforms = []
        self.enemies = []
        self.clones = []
        self.fragments = []
        self.player_bullets = []
        self.enemy_bullets = []
        self.particles = []
        self.bg_lines = []

        # State variables will be properly initialized in reset()
        # self.reset() # reset() is called by the environment runner, no need to call it here.
        
        # --- Self-Validation ---
        # self.validate_implementation() # No need to run this in the final version


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Player State ---
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.player_shoot_cooldown = 0
        self.last_shift_state = 0
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.data_fragments_collected = 0
        self.game_over = False
        self.reward_this_step = 0

        # --- Reset Progression ---
        self.stream_speed = 1.0
        self.enemy_spawn_prob = 0.01
        self.weapon_type = 0  # 0: basic, 1: spread, 2: piercing
        self.clone_fire_rate_multiplier = 1.0

        # --- Clear Dynamic Objects ---
        self.platforms.clear()
        self.enemies.clear()
        self.clones.clear()
        self.fragments.clear()
        self.player_bullets.clear()
        self.enemy_bullets.clear()
        self.particles.clear()

        # --- Procedural Generation ---
        # Initial platforms
        for i in range(15):
            self.platforms.append(pygame.Rect(
                self.np_random.integers(0, self.SCREEN_WIDTH - 80),
                self.np_random.integers(0, self.SCREEN_HEIGHT),
                self.np_random.integers(80, 150),
                18
            ))
        # Ensure a starting platform for the player
        self.platforms.append(pygame.Rect(self.player_pos.x - 40, self.player_pos.y + 40, 80, 18))
        
        # Background data stream lines
        self.bg_lines = [
            (self.np_random.random() * self.SCREEN_WIDTH, self.np_random.random() * self.SCREEN_HEIGHT)
            for _ in range(100)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_this_step = 0.01  # Survival reward

        # --- Handle Input and Cooldowns ---
        self._handle_input(action)
        self.player_shoot_cooldown = max(0, self.player_shoot_cooldown - 1)

        # --- Update Game Logic ---
        self._update_player()
        self._update_platforms()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        self._handle_collisions()
        self._update_progression()

        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not truncated:
            self.reward_this_step = -10.0 # Penalty for losing

        # --- Finalize Step ---
        reward = self.reward_this_step
        if self.score > self.high_score:
            self.reward_this_step += 10.0
            self.high_score = self.score
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # region ############# PRIVATE UPDATE LOGIC #############
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal Movement
        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_SPEED * 0.4
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_SPEED * 0.4

        # Jumping
        if movement == 1 and self.on_ground:
            self.player_vel.y = self.PLAYER_JUMP_POWER
            self.on_ground = False
            # sfx: jump

        # Shooting
        if space_held and self.player_shoot_cooldown == 0:
            self._fire_weapon()
            self.player_shoot_cooldown = self.PLAYER_SHOOT_COOLDOWN
            # sfx: shoot
            # Fire clones
            for clone in self.clones:
                if clone['cooldown'] == 0:
                    self._fire_weapon(clone['pos'])
                    clone['cooldown'] = int(self.PLAYER_SHOOT_COOLDOWN / self.clone_fire_rate_multiplier)

        # Deploy Clone (on key press, not hold)
        if shift_held and not self.last_shift_state and len(self.clones) < self.MAX_CLONES:
            self.clones.append({
                'pos': self.player_pos.copy(),
                'rect': pygame.Rect(self.player_pos, self.CLONE_SIZE),
                'cooldown': 0,
                'spawn_time': self.steps
            })
            # sfx: deploy_clone
        self.last_shift_state = shift_held
        
        # Update clone cooldowns
        for clone in self.clones:
            clone['cooldown'] = max(0, clone['cooldown'] - 1)


    def _update_player(self):
        # Apply physics
        self.player_vel.x *= 0.8  # Damping
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
        self.player_vel.y += self.GRAVITY
        self.player_pos += self.player_vel

        # Boundary checks
        if self.player_pos.x < 0: self.player_pos.x = 0
        if self.player_pos.x > self.SCREEN_WIDTH - self.PLAYER_SIZE[0]:
            self.player_pos.x = self.SCREEN_WIDTH - self.PLAYER_SIZE[0]

        # Platform collision
        player_rect = pygame.Rect(self.player_pos, self.PLAYER_SIZE)
        self.on_ground = False
        for plat in self.platforms:
            if player_rect.colliderect(plat) and self.player_vel.y > 0:
                # Check if player was above the platform in the previous frame
                prev_player_bottom = player_rect.bottom - self.player_vel.y
                if prev_player_bottom <= plat.top:
                    self.player_pos.y = plat.top - self.PLAYER_SIZE[1]
                    self.player_vel.y = self.stream_speed # Stick to platform
                    self.on_ground = True
                    break

    def _update_platforms(self):
        # Scroll and remove old platforms
        self.platforms = [p for p in self.platforms if p.top < self.SCREEN_HEIGHT]
        for plat in self.platforms:
            plat.y += self.stream_speed

        # Spawn new platforms
        if not any(p.top < 20 for p in self.platforms):
            width = self.np_random.integers(70, 140)
            x_pos = self.np_random.integers(0, self.SCREEN_WIDTH - width)
            self.platforms.append(pygame.Rect(x_pos, -20, width, 18))

    def _update_enemies(self):
        # Spawn new enemies
        if self.np_random.random() < self.enemy_spawn_prob:
            x_pos = self.np_random.integers(0, self.SCREEN_WIDTH - self.ENEMY_SIZE[0])
            vel_x = self.np_random.choice([-1.5, 1.5])
            self.enemies.append({
                'rect': pygame.Rect(x_pos, -self.ENEMY_SIZE[1], *self.ENEMY_SIZE),
                'vel': pygame.Vector2(vel_x, 0.5),
                'shoot_cooldown': self.np_random.integers(0, self.ENEMY_SHOOT_COOLDOWN)
            })
            
        # Update existing enemies
        for enemy in self.enemies:
            enemy['rect'].move_ip(enemy['vel'])
            # Bounce off walls
            if enemy['rect'].left < 0 or enemy['rect'].right > self.SCREEN_WIDTH:
                enemy['vel'].x *= -1
            
            # Shooting logic
            enemy['shoot_cooldown'] -= 1
            if enemy['shoot_cooldown'] <= 0:
                # sfx: enemy_shoot
                direction = (self.player_pos - enemy['rect'].center).normalize()
                bullet_vel = direction * 4
                self.enemy_bullets.append({
                    'rect': pygame.Rect(enemy['rect'].centerx, enemy['rect'].centery, 8, 8),
                    'vel': bullet_vel
                })
                enemy['shoot_cooldown'] = self.ENEMY_SHOOT_COOLDOWN + self.np_random.integers(-10, 10)
        
        # Remove off-screen enemies
        self.enemies = [e for e in self.enemies if e['rect'].top < self.SCREEN_HEIGHT]

    def _update_projectiles(self):
        # Player bullets
        for bullet in self.player_bullets:
            bullet['rect'].move_ip(bullet['vel'])
        self.player_bullets = [b for b in self.player_bullets if self.screen.get_rect().colliderect(b['rect'])]
        
        # Enemy bullets
        for bullet in self.enemy_bullets:
            bullet['rect'].move_ip(bullet['vel'])
        self.enemy_bullets = [b for b in self.enemy_bullets if self.screen.get_rect().colliderect(b['rect'])]

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos, self.PLAYER_SIZE)

        # Player bullets vs Enemies
        new_enemies = []
        for enemy in self.enemies:
            hit = False
            bullets_to_remove = []
            for bullet in self.player_bullets:
                if enemy['rect'].colliderect(bullet['rect']):
                    bullets_to_remove.append(bullet)
                    hit = True
                    # If weapon is not piercing, break after one hit
                    if self.weapon_type != 2:
                        break
            
            for bullet in bullets_to_remove:
                if bullet in self.player_bullets:
                    self.player_bullets.remove(bullet)

            if hit:
                self.score += 10
                self.reward_this_step += 1.0
                self._spawn_explosion(enemy['rect'].center, 15, self.COLOR_PARTICLE_EXPLOSION)
                # sfx: explosion
                # Spawn data fragments on kill
                if self.np_random.random() < 0.5:
                    self.fragments.append(pygame.Rect(enemy['rect'].centerx, enemy['rect'].centery, 12, 12))
            else:
                new_enemies.append(enemy)
        self.enemies = new_enemies

        # Player vs Fragments
        new_fragments = []
        for frag in self.fragments:
            frag.y += self.stream_speed
            if player_rect.colliderect(frag):
                self.data_fragments_collected += 1
                self.score += 5
                self.reward_this_step += 0.1
                # sfx: collect_fragment
            else:
                new_fragments.append(frag)
        self.fragments = [f for f in new_fragments if f.top < self.SCREEN_HEIGHT]

        # Player vs Enemies or Enemy bullets
        for enemy in self.enemies:
            if player_rect.colliderect(enemy['rect']):
                self.game_over = True
        for bullet in self.enemy_bullets:
            if player_rect.colliderect(bullet['rect']):
                self.game_over = True
        
        if self.game_over:
            self._spawn_explosion(player_rect.center, 30, self.COLOR_PLAYER)
            # sfx: player_death

    def _update_progression(self):
        # Increase difficulty over time
        if self.steps > 0 and self.steps % 500 == 0:
            self.stream_speed = min(3.0, self.stream_speed + 0.05)
        if self.steps > 0 and self.steps % 250 == 0:
            self.enemy_spawn_prob = min(0.05, self.enemy_spawn_prob + 0.02)
            
        # Unlock weapons based on score
        if self.score >= 500 and self.weapon_type == 0:
            self.weapon_type = 1 # Spread shot
        if self.score >= 1500 and self.weapon_type == 1:
            self.weapon_type = 2 # Piercing shot
            
        # Upgrade clones based on score
        if self.score >= 1000 and self.clone_fire_rate_multiplier == 1.0:
            self.clone_fire_rate_multiplier = 1.5
        if self.score >= 2000 and self.clone_fire_rate_multiplier == 1.5:
            self.clone_fire_rate_multiplier = 2.0
            
    def _check_termination(self):
        player_off_screen = self.player_pos.y > self.SCREEN_HEIGHT
        if player_off_screen:
            self.game_over = True

        return self.game_over
    # endregion

    # region ############# FIRING & EFFECTS #############
    def _fire_weapon(self, origin=None):
        if origin is None:
            origin = pygame.Vector2(self.player_pos.x + self.PLAYER_SIZE[0] / 2, self.player_pos.y + self.PLAYER_SIZE[1] / 2)
        
        if self.weapon_type == 0: # Basic
            self.player_bullets.append({'rect': pygame.Rect(origin.x-3, origin.y-3, 6, 6), 'vel': pygame.Vector2(0, -8)})
        elif self.weapon_type == 1: # Spread
            self.player_bullets.append({'rect': pygame.Rect(origin.x-3, origin.y-3, 5, 5), 'vel': pygame.Vector2(0, -8)})
            self.player_bullets.append({'rect': pygame.Rect(origin.x-3, origin.y-3, 5, 5), 'vel': pygame.Vector2(-2, -7)})
            self.player_bullets.append({'rect': pygame.Rect(origin.x-3, origin.y-3, 5, 5), 'vel': pygame.Vector2(2, -7)})
        elif self.weapon_type == 2: # Piercing (visual only, logic in collision)
             self.player_bullets.append({'rect': pygame.Rect(origin.x-2, origin.y-5, 4, 10), 'vel': pygame.Vector2(0, -10)})

    def _spawn_explosion(self, position, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 4 + 1
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(position),
                'vel': vel,
                'radius': self.np_random.random() * 4 + 2,
                'color': color,
                'lifespan': self.np_random.integers(20, 40)
            })
    # endregion

    # region ############# GYM & PYGAME INTERFACE #############
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "data_fragments": self.data_fragments_collected,
            "weapon_type": self.weapon_type,
            "clones": len(self.clones)
        }

    def _render_background(self):
        # Gradient background
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Scrolling data lines
        new_lines = []
        for x, y in self.bg_lines:
            new_y = y + self.stream_speed * 0.5
            if new_y > self.SCREEN_HEIGHT:
                new_y = 0
                x = self.np_random.random() * self.SCREEN_WIDTH
            pygame.draw.line(self.screen, (50, 20, 80), (x, new_y), (x, new_y + 10), 1)
            new_lines.append((x, new_y))
        self.bg_lines = new_lines

    def _render_game(self):
        # Platforms
        for plat in self.platforms:
            pygame.gfxdraw.box(self.screen, plat, (*self.COLOR_PLATFORM_GLOW, 150))
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, 2)

        # Fragments
        for frag in self.fragments:
            pygame.draw.rect(self.screen, self.COLOR_FRAGMENT, frag)
            pygame.gfxdraw.rectangle(self.screen, frag, (*self.COLOR_FRAGMENT, 100))

        # Clones
        for clone in self.clones:
            clone_rect = pygame.Rect(int(clone['pos'].x), int(clone['pos'].y), *self.CLONE_SIZE)
            glow_rect = clone_rect.inflate(10, 10)
            pygame.gfxdraw.box(self.screen, glow_rect, (*self.COLOR_PLAYER_GLOW, 50))
            pygame.draw.rect(self.screen, self.COLOR_CLONE, clone_rect, 0, border_radius=3)
            
        # Player
        if not self.game_over:
            player_rect = pygame.Rect(int(self.player_pos.x), int(self.player_pos.y), *self.PLAYER_SIZE)
            glow_rect = player_rect.inflate(15, 15)
            pygame.gfxdraw.box(self.screen, glow_rect, (*self.COLOR_PLAYER_GLOW, 100))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, 0, border_radius=3)

        # Enemies
        for enemy in self.enemies:
            glow_rect = enemy['rect'].inflate(10, 10)
            pygame.gfxdraw.box(self.screen, glow_rect, (*self.COLOR_ENEMY_GLOW, 100))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, enemy['rect'], 0, border_radius=3)

        # Projectiles
        for bullet in self.player_bullets:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_BULLET, bullet['rect'])
        for bullet in self.enemy_bullets:
            pygame.gfxdraw.filled_circle(self.screen, bullet['rect'].centerx, bullet['rect'].centery, 4, self.COLOR_ENEMY_BULLET)
            
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['lifespan'] * 6)))
            try:
                # Use a surface for alpha blending
                s = pygame.Surface((int(p['radius'])*2, int(p['radius'])*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(s, int(p['radius']), int(p['radius']), int(p['radius']), (*p['color'], alpha))
                self.screen.blit(s, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))
            except (pygame.error, ValueError):
                # Skip particles with invalid radius
                pass

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Data Fragments
        frag_surf = self.font_ui.render(f"FRAGMENTS: {self.data_fragments_collected}", True, self.COLOR_TEXT)
        self.screen.blit(frag_surf, (self.SCREEN_WIDTH - frag_surf.get_width() - 10, 10))

        # Weapon display
        weapon_names = {0: "BASIC", 1: "SPREAD", 2: "PIERCER"}
        weapon_colors = {0: self.COLOR_PLAYER_BULLET, 1: (255, 165, 0), 2: (255, 0, 255)}
        weapon_text = f"WEAPON: {weapon_names[self.weapon_type]}"
        weapon_surf = self.font_weapon.render(weapon_text, True, weapon_colors[self.weapon_type])
        self.screen.blit(weapon_surf, (self.SCREEN_WIDTH / 2 - weapon_surf.get_width() / 2, self.SCREEN_HEIGHT - 30))

    def close(self):
        pygame.quit()

# endregion

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For manual play, we need a real display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cyberpunk Platformer")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not terminated and not truncated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        # elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2 # Not used
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()
    pygame.quit()