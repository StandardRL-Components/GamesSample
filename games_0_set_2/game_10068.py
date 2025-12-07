import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:45:36.782408
# Source Brief: brief_00068.md
# Brief Index: 68
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete


# Helper class for the Player
class Player:
    def __init__(self):
        self.w = 30
        self.h = 40
        self.reset()

    def reset(self):
        self.x = 100
        self.y = 200
        self.vx = 0
        self.vy = 0
        self.on_ground = False
        self.powerups = {} # e.g. {"double_jump": 100, "invincible": 50}
        self.has_double_jumped = False

    def draw(self, surface, camera_x):
        # Body
        player_rect = pygame.Rect(int(self.x - camera_x), int(self.y - self.h), self.w, self.h)
        body_color = (255, 255, 0) if 'invincible' not in self.powerups else (255, 128, 0)
        pygame.draw.rect(surface, body_color, player_rect, border_radius=8)
        
        # Glow effect
        glow_color = body_color
        for i in range(10, 0, -1):
            alpha = 100 - i * 10
            # Pygame.gfxdraw.aacircle requires a tuple of 4 for color with alpha
            pygame.gfxdraw.aacircle(surface, int(self.x - camera_x + self.w / 2), int(self.y - self.h / 2), 20 + i, glow_color + (alpha,))
            
        # Eyes
        eye_y = self.y - self.h * 0.6
        eye_x_offset = self.w * 0.2
        pygame.draw.circle(surface, (0,0,0), (int(self.x - camera_x + eye_x_offset), int(eye_y)), 3)
        pygame.draw.circle(surface, (0,0,0), (int(self.x - camera_x + self.w - eye_x_offset), int(eye_y)), 3)

    def update(self, movement_action, gravity):
        # Horizontal movement
        speed_multiplier = 1.5 if 'speed_boost' in self.powerups else 1.0
        target_vx = 0
        if movement_action == 3: # Left
            target_vx = -5 * speed_multiplier
        elif movement_action == 4: # Right
            target_vx = 5 * speed_multiplier
        
        # Smooth acceleration/deceleration
        self.vx += (target_vx - self.vx) * 0.2
        self.x += self.vx

        # Vertical movement (gravity)
        if not self.on_ground:
            self.vy += gravity
            if movement_action == 2: # Fast fall
                self.vy += gravity * 1.5 
            self.y += self.vy

        # Update power-ups timers
        for powerup in list(self.powerups.keys()):
            self.powerups[powerup] -= 1
            if self.powerups[powerup] <= 0:
                del self.powerups[powerup]
        
        self.on_ground = False

    def jump(self):
        if self.on_ground:
            self.vy = -12 # Jump strength
            self.on_ground = False
            self.has_double_jumped = False
            # Sound: Player Jump
            return True
        elif 'double_jump' in self.powerups and not self.has_double_jumped:
            self.vy = -10 # Double jump strength
            self.has_double_jumped = True
            # Sound: Double Jump
            return True
        return False

# Helper class for Particles
class Particle:
    def __init__(self, x, y, color, life, size_range=(2, 5), speed_range=(-2, 2)):
        self.x = x
        self.y = y
        self.vx = random.uniform(speed_range[0], speed_range[1])
        self.vy = random.uniform(speed_range[0], speed_range[1])
        self.color = color
        self.life = life
        self.max_life = life
        self.size = random.uniform(size_range[0], size_range[1])

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1 # particle gravity
        self.life -= 1

    def draw(self, surface, camera_x):
        if self.life > 0:
            lerp_factor = self.life / self.max_life
            current_size = int(self.size * lerp_factor)
            if current_size > 0:
                color_with_alpha = self.color + (int(255 * lerp_factor),)
                temp_surf = pygame.Surface((current_size*2, current_size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color_with_alpha, (current_size, current_size), current_size)
                surface.blit(temp_surf, (int(self.x - camera_x - current_size), int(self.y - current_size)))

# Helper class for Rhythm Challenge
class RhythmChallenge:
    def __init__(self, difficulty):
        self.note_count = 3 + int(difficulty)
        self.sequence = [random.randint(0, 1) for _ in range(self.note_count)]
        self.notes = []
        self.hit_zone_y = 200
        self.note_speed = 5
        self.current_note_index = 0
        self.state = 'active' # 'active', 'success', 'fail'
        self.feedback_timer = 0
        self.input_grace_period = 10 # frames
        
        for i, note_type in enumerate(self.sequence):
            y_pos = self.hit_zone_y - (i + 2) * 60
            x_pos = 280 if note_type == 0 else 360
            self.notes.append(pygame.Rect(x_pos, y_pos, 40, 20))

    def update(self, space_pressed, shift_pressed):
        if self.state != 'active':
            self.feedback_timer -= 1
            if self.feedback_timer <= 0:
                return True # Challenge is complete
            return False

        # Move notes down
        for note in self.notes:
            note.y += self.note_speed

        if self.current_note_index < len(self.sequence):
            active_note = self.notes[self.current_note_index]
            note_type = self.sequence[self.current_note_index]
            
            # Check for miss
            if active_note.top > self.hit_zone_y + self.input_grace_period:
                self.state = 'fail'
                self.feedback_timer = 30
                # Sound: Rhythm Fail
                return False

            # Check for hit
            in_hit_zone = abs(active_note.centery - self.hit_zone_y) < self.input_grace_period
            
            correct_input = (note_type == 0 and space_pressed) or (note_type == 1 and shift_pressed)
            wrong_input = (note_type == 1 and space_pressed) or (note_type == 0 and shift_pressed)

            if in_hit_zone and correct_input:
                self.current_note_index += 1
                # Sound: Rhythm Hit
                if self.current_note_index == len(self.sequence):
                    self.state = 'success'
                    self.feedback_timer = 30
            elif (in_hit_zone and wrong_input) or (not in_hit_zone and (space_pressed or shift_pressed)):
                # Fail if wrong button is pressed in zone, or any button is pressed outside zone
                self.state = 'fail'
                self.feedback_timer = 30
                # Sound: Rhythm Fail
        
        return False

    def draw(self, surface):
        # Draw backdrop
        s = pygame.Surface((200, 400), pygame.SRCALPHA)
        s.fill((0, 0, 50, 150))
        surface.blit(s, (220, 0))

        # Draw hit zone
        pygame.draw.rect(surface, (0, 255, 255, 100), (220, self.hit_zone_y - self.input_grace_period, 200, self.input_grace_period*2), border_radius=5)
        pygame.draw.line(surface, (255, 255, 255), (220, self.hit_zone_y), (420, self.hit_zone_y), 2)

        # Draw note tracks
        pygame.draw.line(surface, (255, 255, 255, 50), (300, 0), (300, 400), 1)
        pygame.draw.line(surface, (255, 255, 255, 50), (340, 0), (340, 400), 1)

        # Draw notes
        for i in range(self.current_note_index, len(self.sequence)):
            note_rect = self.notes[i]
            note_type = self.sequence[i]
            color = (100, 255, 100) if note_type == 0 else (255, 100, 255) # Green for space, Magenta for shift
            pygame.draw.rect(surface, color, note_rect, border_radius=5)

        # Draw feedback
        if self.state == 'success':
            font = pygame.font.SysFont('Consolas', 40, bold=True)
            text = font.render("SUCCESS!", True, (0, 255, 0))
            surface.blit(text, (320 - text.get_width() // 2, 100))
        elif self.state == 'fail':
            font = pygame.font.SysFont('Consolas', 40, bold=True)
            text = font.render("FAIL!", True, (255, 0, 0))
            surface.blit(text, (320 - text.get_width() // 2, 100))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A side-scrolling platformer where you run, jump, and complete rhythm challenges to gain power-ups and reach the finish line."
    )
    user_guide = (
        "Controls: ←→ to run, ↑ to jump, and ↓ to fall faster. During rhythm challenges, press space or shift to hit the notes."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_LENGTH = 15000
        self.MAX_STEPS = 5000
        self.GRAVITY = 0.6
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('Consolas', 18)
        self.font_large = pygame.font.SysFont('Consolas', 32, bold=True)
        
        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_PLATFORM = (0, 80, 120)
        self.COLOR_STAFF_LINES = (20, 40, 60)
        self.COLOR_FINISH = (0, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        
        # Game State - initialized in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.player = None
        self.platforms = None
        self.rhythm_triggers = None
        self.particles = None
        self.camera_x = None
        self.last_player_x = None
        self.game_mode = None # 'RUNNING' or 'RHYTHM'
        self.rhythm_challenge = None
        self.consecutive_rhythm_fails = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.reward_this_step = None

        # Call reset to initialize all state variables
        self.reset()

    def _generate_world(self):
        self.platforms = []
        self.rhythm_triggers = []
        
        # Initial platform
        current_x = -200
        self.platforms.append(pygame.Rect(current_x, self.HEIGHT - 50, 600, 100))
        current_x += 600
        
        # Procedural generation loop
        rhythm_trigger_dist = 2000
        next_rhythm_trigger = rhythm_trigger_dist

        while current_x < self.WORLD_LENGTH:
            # Difficulty scaling based on progress
            progress_ratio = current_x / self.WORLD_LENGTH
            max_gap = 100 + 150 * progress_ratio
            min_plat = 400 - 300 * progress_ratio
            max_plat = 800 - 500 * progress_ratio
            
            gap = random.uniform(50, max_gap)
            current_x += gap
            
            plat_length = random.uniform(min_plat, max_plat)
            plat_height = self.HEIGHT - 50 + random.uniform(-80 * progress_ratio, 80 * progress_ratio)
            self.platforms.append(pygame.Rect(current_x, plat_height, plat_length, 100))
            
            if current_x > next_rhythm_trigger:
                self.rhythm_triggers.append(current_x)
                next_rhythm_trigger += rhythm_trigger_dist * random.uniform(0.8, 1.2)

            current_x += plat_length

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_x = 0
        self.consecutive_rhythm_fails = 0
        self.game_mode = 'RUNNING'
        self.rhythm_challenge = None
        self.reward_this_step = 0
        
        self.player = Player()
        self.player.reset()
        self.last_player_x = self.player.x
        
        self._generate_world()
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        self.reward_this_step = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Game Logic ---
        if self.game_mode == 'RUNNING':
            self._update_running(movement, space_pressed)
        elif self.game_mode == 'RHYTHM':
            self._update_rhythm(space_pressed, shift_pressed)
        
        # --- Universal Updates ---
        self.steps += 1
        
        # Update particles
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)
        
        # Update camera
        self.camera_x = self.player.x - 150
        
        # Check termination conditions
        terminated = self._check_termination()
        
        self.score += self.reward_this_step

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        self.last_player_x = self.player.x

        # The truncated flag is always False because termination is handled internally.
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Gymnasium standard is to set both to True on timeout

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_running(self, movement, space_pressed):
        # Update player based on action
        if movement == 1: # Jump
            if self.player.jump():
                for _ in range(20):
                    self.particles.append(Particle(self.player.x + self.player.w/2, self.player.y, (200, 200, 255), 30))

        self.player.update(movement, self.GRAVITY)
        
        # Forward progress reward
        self.reward_this_step += (self.player.x - self.last_player_x) * 0.01

        # Collision detection with platforms
        player_rect = pygame.Rect(self.player.x, self.player.y - self.player.h, self.player.w, self.player.h)
        self.player.on_ground = False
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # Check if landing on top
                if self.player.vy > 0 and player_rect.bottom < plat.top + self.player.vy + 1:
                    self.player.y = plat.top
                    self.player.vy = 0
                    if not self.player.on_ground: # First frame on ground
                        for _ in range(5):
                            self.particles.append(Particle(self.player.x + self.player.w/2, self.player.y, (200,200,200), 20))
                    self.player.on_ground = True
                    self.player.has_double_jumped = False
                # Check for hitting from below
                elif self.player.vy < 0 and player_rect.top > plat.bottom + self.player.vy -1:
                    self.player.vy = 0
                # Horizontal collision
                else:
                    if self.player.vx > 0: # Moving right
                        self.player.x = plat.left - self.player.w
                    elif self.player.vx < 0: # Moving left
                        self.player.x = plat.right
                    self.player.vx = 0

        # Check for rhythm challenge trigger
        if self.rhythm_triggers and self.player.x > self.rhythm_triggers[0]:
            self.rhythm_triggers.pop(0)
            self.game_mode = 'RHYTHM'
            difficulty = self.player.x / self.WORLD_LENGTH * 3
            self.rhythm_challenge = RhythmChallenge(difficulty)
            # Sound: Rhythm Challenge Start
    
    def _update_rhythm(self, space_pressed, shift_pressed):
        if self.rhythm_challenge:
            is_done = self.rhythm_challenge.update(space_pressed, shift_pressed)
            
            # Create feedback particles
            if space_pressed:
                for _ in range(10): self.particles.append(Particle(300, self.rhythm_challenge.hit_zone_y, (100, 255, 100), 15))
            if shift_pressed:
                for _ in range(10): self.particles.append(Particle(340, self.rhythm_challenge.hit_zone_y, (255, 100, 255), 15))

            if is_done:
                if self.rhythm_challenge.state == 'success':
                    self.reward_this_step += 5 # Rhythm success reward
                    self.consecutive_rhythm_fails = 0
                    
                    # Grant a random powerup
                    powerup_type = random.choice(['double_jump', 'speed_boost', 'invincible'])
                    self.player.powerups[powerup_type] = 300 # 10 seconds at 30fps
                    self.reward_this_step += 10 # Powerup collection reward

                elif self.rhythm_challenge.state == 'fail':
                    self.reward_this_step -= 2 # Rhythm fail penalty
                    self.consecutive_rhythm_fails += 1
                
                self.rhythm_challenge = None
                self.game_mode = 'RUNNING'

    def _check_termination(self):
        if self.game_over:
            return True
            
        # Fall off map
        if self.player.y > self.HEIGHT + 100:
            self.game_over = True
            self.reward_this_step -= 50
            return True
            
        # Reach finish line
        if self.player.x > self.WORLD_LENGTH:
            self.game_over = True
            self.reward_this_step += 100
            return True

        # Too many rhythm fails
        if self.consecutive_rhythm_fails >= 3:
            self.game_over = True
            self.reward_this_step -= 50
            return True

        # Max steps is handled as truncation
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Parallax staff lines
        for i in range(5):
            y = self.HEIGHT * 0.4 + i * 20
            # Slower scroll rate for parallax effect
            start_x = - (self.camera_x * 0.5 % 50)
            for j in range(self.WIDTH // 50 + 2):
                x = start_x + j * 50
                pygame.draw.line(self.screen, self.COLOR_STAFF_LINES, (x, y), (x + 30, y), 2)

    def _render_game(self):
        # Draw platforms
        for plat in self.platforms:
            if plat.right > self.camera_x and plat.left < self.camera_x + self.WIDTH:
                draw_rect = plat.move(-self.camera_x, 0)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, draw_rect)
        
        # Draw finish line
        finish_x = self.WORLD_LENGTH - self.camera_x
        if finish_x < self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_x, 0), (finish_x, self.HEIGHT), 5)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen, self.camera_x)

        # Draw player
        self.player.draw(self.screen, self.camera_x)

        # Draw rhythm game overlay
        if self.game_mode == 'RHYTHM' and self.rhythm_challenge:
            self.rhythm_challenge.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Progress Bar
        progress = max(0, min(1, self.player.x / self.WORLD_LENGTH))
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, (50, 50, 80), (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, (100, 200, 255), (10, 10, int(bar_width * progress), bar_height))
        
        # Power-ups
        y_offset = 40
        for powerup, duration in self.player.powerups.items():
            text = self.font_small.render(f"{powerup.replace('_',' ').upper()}: {duration//30}s", True, (255, 255, 100))
            self.screen.blit(text, (10, y_offset))
            y_offset += 20
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_x": self.player.x,
            "progress": self.player.x / self.WORLD_LENGTH,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and visualization, and will not be run by the evaluator.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Rhythm Runner")
    clock = pygame.time.Clock()
    
    while running:
        movement_action = 0 # none
        space_action = 0 # released
        shift_action = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        if keys[pygame.K_DOWN]:
            movement_action = 2
        if keys[pygame.K_LEFT]:
            movement_action = 3
        if keys[pygame.K_RIGHT]:
            movement_action = 4
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward}, Score: {info['score']}, Steps: {info['steps']}")
            total_reward = 0
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS for human play
        
    env.close()