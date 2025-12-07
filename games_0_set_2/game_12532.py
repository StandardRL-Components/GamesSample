import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:12:29.132868
# Source Brief: brief_02532.md
# Brief Index: 2532
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Rhythm Jumper: A rhythm-based action game where players must jump on the beat
    to collect notes, craft power-ups, and evade guards.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Rhythm Jumper is a rhythm-based action game. Jump on the beat to collect notes, "
        "craft power-ups, and evade the guards' watchful eyes."
    )
    user_guide = (
        "Use the ←→ arrow keys to move and press space to jump. "
        "Time your jumps with the beat to earn bonuses and remain undetected."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30  # Assumed FPS for smooth interpolation

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 60)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 128, 128)
    COLOR_GUARD = (255, 100, 0)
    COLOR_GUARD_GLOW = (180, 70, 0)
    COLOR_VISION_CONE = (255, 100, 0, 50) # RGBA
    NOTE_COLORS = {
        "speed": (255, 0, 128),   # Red -> Magenta for better contrast
        "stealth": (0, 128, 255), # Blue
        "score": (50, 255, 50)    # Green
    }
    COLOR_UI_TEXT = (220, 220, 240)

    # Game Parameters
    PLAYER_X_SPEED = 6
    PLAYER_JUMP_FORCE = -12
    GRAVITY = 0.6
    MAX_STEPS = 2400 # 80 seconds at 30 FPS
    BEAT_TIMING_WINDOW = 0.15  # +/- 15% of beat duration is "on-beat"
    POWERUP_NOTE_REQUIREMENT = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_state = 0
        self.last_jump_info = None # Stores (x, y, on_beat) for guard detection

        self.player = None
        self.guards = []
        self.notes = []
        self.particles = []

        self.tempo = 0
        self.seconds_per_beat = 0
        self.beat_progress = 0
        self.world_scroll_speed = 0

        self.note_counts = {}
        self.powerup_timers = {}
        self.double_jump_unlocked = False
        
        self.reward_this_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.last_space_state = 0
        self.last_jump_info = None
        
        # Player
        self.player = Player(self.WIDTH / 2, self.HEIGHT * 0.75)

        # Lists
        self.guards = []
        self.notes = []
        self.particles = []

        # Rhythm
        self.tempo = 120.0
        self._update_tempo_dependencies()
        self.beat_progress = 0

        # Powerups
        self.note_counts = { "speed": 0, "stealth": 0, "score": 0 }
        self.powerup_timers = { "speed": 0, "stealth": 0, "score_multiplier": 0 }
        self.double_jump_unlocked = False

        # Initial Population
        self._spawn_initial_guards()
        for _ in range(10):
            self._spawn_note()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_this_step = 0 # Reset per-step reward
        
        self._handle_input(action)
        self._update_game_state()
        self._check_collisions_and_detections()
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        reward = self.reward_this_step
        if terminated:
            if self.game_over: # Lost
                reward += -50.0
            elif self.steps >= self.MAX_STEPS: # Won
                reward += 100.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Horizontal Movement
        if movement == 3: # Left
            self.player.vx = -self.PLAYER_X_SPEED
        elif movement == 4: # Right
            self.player.vx = self.PLAYER_X_SPEED
        else:
            self.player.vx = 0

        # Jump (on rising edge of space bar)
        if space_held and not self.last_space_state:
            time_from_beat = min(self.beat_progress, self.seconds_per_beat - self.beat_progress)
            is_on_beat = time_from_beat / self.seconds_per_beat < self.BEAT_TIMING_WINDOW
            
            if self.player.jump(is_on_beat, self.double_jump_unlocked):
                # SFX: Jump_SFX() or Double_Jump_SFX()
                self._spawn_particles(self.player.x, self.player.y + self.player.h / 2, self.COLOR_PLAYER, 20)
                if not is_on_beat:
                    self.reward_this_step -= 0.1
                self.last_jump_info = (self.player.x, self.player.y, is_on_beat)
        
        self.last_space_state = space_held

    def _update_game_state(self):
        # --- Update Time and Rhythm ---
        dt = 1.0 / self.FPS
        self.beat_progress += dt
        if self.beat_progress >= self.seconds_per_beat:
            self.beat_progress -= self.seconds_per_beat
            self._spawn_note() # Spawn a note on the beat
            # SFX: Metronome_Tick_SFX()

        # --- Difficulty Progression ---
        if self.steps > 0:
            if self.steps % (20 * self.FPS) == 0: # Every 20 seconds
                self.tempo += 5
                self._update_tempo_dependencies()
                for g in self.guards: g.speed += 0.05
            if self.steps % (30 * self.FPS) == 0: # Every 30 seconds
                for g in self.guards: g.cone_angle = max(30, g.cone_angle - 1)
            if self.steps == (60 * self.FPS): # At 60 seconds
                self.double_jump_unlocked = True

        # --- Update Game Objects ---
        # Apply powerup effects
        speed_mult = 1.5 if self.powerup_timers['speed'] > 0 else 1.0
        self.player.update(self.GRAVITY, speed_mult, self.WIDTH, self.HEIGHT * 0.75)
        
        # Scroll world
        for obj_list in [self.notes, self.guards, self.particles]:
            for obj in obj_list:
                obj.y += self.world_scroll_speed
        
        # Update individual objects
        for guard in self.guards: guard.update()
        for particle in self.particles: particle.update()

        # --- Manage Object Lists ---
        self.notes = [n for n in self.notes if n.y < self.HEIGHT + 20]
        self.guards = [g for g in self.guards if g.y < self.HEIGHT + 50]
        self.particles = [p for p in self.particles if p.lifespan > 0]
        
        if len(self.guards) < 2: # Ensure there are always guards
             self._spawn_guard()

        # Decrement powerup timers
        for key in self.powerup_timers:
            self.powerup_timers[key] = max(0, self.powerup_timers[key] - 1)

    def _check_collisions_and_detections(self):
        player_rect = self.player.get_rect()

        # Player -> Notes
        for note in self.notes[:]:
            if player_rect.colliderect(note.get_rect()):
                self.notes.remove(note)
                # SFX: Note_Collect_SFX()
                self._spawn_particles(note.x, note.y, self.NOTE_COLORS[note.type], 15)
                
                multiplier = 2 if self.powerup_timers['score_multiplier'] > 0 else 1
                self.reward_this_step += 1.0 * multiplier
                self.score += 10 * multiplier

                # Powerup crafting
                self.note_counts[note.type] += 1
                if self.note_counts[note.type] >= self.POWERUP_NOTE_REQUIREMENT:
                    self.note_counts[note.type] = 0
                    self.reward_this_step += 5.0
                    # SFX: Powerup_Craft_SFX()
                    if note.type == "speed": self.powerup_timers["speed"] = 10 * self.FPS
                    elif note.type == "stealth": self.powerup_timers["stealth"] = 10 * self.FPS
                    elif note.type == "score": self.powerup_timers["score_multiplier"] = 15 * self.FPS


        # Player Jump -> Guard Detection
        if self.last_jump_info:
            jump_x, jump_y, on_beat = self.last_jump_info
            if not on_beat:
                for guard in self.guards:
                    stealth_active = self.powerup_timers['stealth'] > 0
                    if guard.detects(jump_x, jump_y, stealth_active):
                        self.game_over = True
                        # SFX: Detection_SFX()
                        self._spawn_particles(self.player.x, self.player.y, self.COLOR_GUARD, 50, -2)
                        break
            self.last_jump_info = None # Consume the jump event

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # --- Game Elements ---
        for p in self.particles: p.draw(self.screen)
        for n in self.notes: n.draw(self.screen)
        for g in self.guards: g.draw(self.screen, self.powerup_timers['stealth'] > 0)
        self.player.draw(self.screen)
        self._render_beat_indicator()
        
        # --- UI Overlay ---
        self._render_ui()

        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Powerup Status
        y_offset = 40
        for p_type, timer in self.powerup_timers.items():
            if timer > 0:
                color = self.NOTE_COLORS.get(p_type.replace('_multiplier', ''), (255,255,255))
                text = self.font_small.render(f"{p_type.upper()} ACTIVE", True, color)
                self.screen.blit(text, (10, y_offset))
                
                bar_width = (timer / (15 * self.FPS)) * 100
                pygame.draw.rect(self.screen, color, (10, y_offset + 20, bar_width, 5))
                y_offset += 35
        
        # Double Jump Indicator
        if self.double_jump_unlocked:
            dj_color = self.COLOR_PLAYER if self.player.can_double_jump and self.player.is_jumping else (100,100,100)
            dj_text = self.font_small.render("DOUBLE JUMP READY", True, dj_color)
            self.screen.blit(dj_text, (self.WIDTH - dj_text.get_width() - 10, 10))

    def _render_beat_indicator(self):
        progress = self.beat_progress / self.seconds_per_beat
        # A pulse that is sharp at the beat and fades
        pulse = (1 - progress)**4
        
        radius = int(10 + 15 * pulse)
        alpha = int(50 + 200 * pulse)
        color = (*self.COLOR_PLAYER, alpha)
        
        # Use a temporary surface for transparency
        temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color)
        pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius, color)
        self.screen.blit(temp_surf, (int(self.player.x - radius), int(self.player.y - radius)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tempo": self.tempo,
            "powerups_active": sum(1 for t in self.powerup_timers.values() if t > 0)
        }

    def _update_tempo_dependencies(self):
        self.seconds_per_beat = 60.0 / self.tempo
        # Scroll one screen height every 8 beats
        self.world_scroll_speed = self.HEIGHT / (self.seconds_per_beat * 8 * self.FPS)

    def _spawn_note(self):
        # Spawn notes just off the top of the screen
        note_type = self.np_random.choice(list(self.NOTE_COLORS.keys()))
        x = self.np_random.uniform(20, self.WIDTH - 20)
        y = -20
        self.notes.append(Note(x, y, note_type))
        
    def _spawn_initial_guards(self):
        self._spawn_guard(y_pos=self.HEIGHT * 0.2)
        self._spawn_guard(y_pos=self.HEIGHT * -0.5) # Off-screen start

    def _spawn_guard(self, y_pos=None):
        if y_pos is None:
            y_pos = -50
        path_width = self.np_random.uniform(self.WIDTH * 0.3, self.WIDTH * 0.7)
        path_center = self.np_random.uniform(path_width/2, self.WIDTH - path_width/2)
        start_x = path_center - path_width / 2
        end_x = path_center + path_width / 2
        self.guards.append(Guard(start_x, end_x, y_pos))

    def _spawn_particles(self, x, y, color, count, speed_mult=1):
        for _ in range(count):
            self.particles.append(Particle(x, y, color, self.np_random, speed_mult))

    def close(self):
        pygame.quit()

# --- Helper Classes ---

class Player:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vx, self.vy = 0, 0
        self.w, self.h = 20, 30
        self.is_jumping = False
        self.can_double_jump = False
        self.base_y = y

    def get_rect(self):
        return pygame.Rect(self.x - self.w / 2, self.y - self.h / 2, self.w, self.h)

    def jump(self, on_beat, double_jump_unlocked):
        jump_force = GameEnv.PLAYER_JUMP_FORCE * (1.5 if on_beat else 1.0)
        if not self.is_jumping:
            self.vy = jump_force
            self.is_jumping = True
            self.can_double_jump = True
            return True
        elif double_jump_unlocked and self.can_double_jump:
            self.vy = jump_force * 0.8 # Double jump is slightly weaker
            self.can_double_jump = False
            return True
        return False

    def update(self, gravity, speed_mult, screen_width, base_y):
        self.base_y = base_y
        self.vy += gravity
        self.x += self.vx * speed_mult
        self.y += self.vy

        # Keep player within horizontal bounds
        self.x = max(self.w / 2, min(self.x, screen_width - self.w / 2))

        # Check for landing
        if self.y >= self.base_y and self.vy > 0:
            self.y = self.base_y
            self.vy = 0
            self.is_jumping = False
            self.can_double_jump = False

    def draw(self, screen):
        rect = self.get_rect()
        # Glow
        glow_rect = rect.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, (*GameEnv.COLOR_PLAYER_GLOW, 100), glow_surf.get_rect())
        screen.blit(glow_surf, glow_rect.topleft)
        # Body
        pygame.draw.ellipse(screen, GameEnv.COLOR_PLAYER, rect)

class Guard:
    def __init__(self, path_start_x, path_end_x, y):
        self.x, self.y = path_start_x, y
        self.path_start, self.path_end = path_start_x, path_end_x
        self.w, self.h = 25, 25
        self.speed = 1.0
        self.direction = 1
        
        self.cone_radius = 200
        self.cone_angle = 70 # degrees
        self.cone_dir_angle = 90 # degrees, pointing down
        self.cone_rotation_speed = 0.5 # degrees per frame

    def get_rect(self):
        return pygame.Rect(self.x - self.w / 2, self.y - self.h / 2, self.w, self.h)

    def update(self):
        self.x += self.speed * self.direction
        if self.x > self.path_end or self.x < self.path_start:
            self.direction *= -1
        
        self.cone_dir_angle += self.cone_rotation_speed * self.direction

    def detects(self, px, py, stealth_active):
        dx, dy = px - self.x, py - self.y
        dist = math.hypot(dx, dy)
        
        cone_angle = self.cone_angle / 2 if stealth_active else self.cone_angle
        cone_radius = self.cone_radius * 0.6 if stealth_active else self.cone_radius

        if dist > cone_radius:
            return False

        angle_to_player = math.degrees(math.atan2(dy, dx))
        angle_diff = (angle_to_player - self.cone_dir_angle + 180) % 360 - 180
        
        return abs(angle_diff) < cone_angle / 2

    def draw(self, screen, stealth_active):
        # Vision Cone
        cone_angle_rad = math.radians(self.cone_angle / 2 if stealth_active else self.cone_angle)
        cone_dir_rad = math.radians(self.cone_dir_angle)
        cone_radius = self.cone_radius * 0.6 if stealth_active else self.cone_radius

        p1 = (int(self.x), int(self.y))
        p2 = (int(self.x + cone_radius * math.cos(cone_dir_rad - cone_angle_rad / 2)),
              int(self.y + cone_radius * math.sin(cone_dir_rad - cone_angle_rad / 2)))
        p3 = (int(self.x + cone_radius * math.cos(cone_dir_rad + cone_angle_rad / 2)),
              int(self.y + cone_radius * math.sin(cone_dir_rad + cone_angle_rad / 2)))
        
        pygame.gfxdraw.aapolygon(screen, [p1, p2, p3], GameEnv.COLOR_VISION_CONE)
        pygame.gfxdraw.filled_polygon(screen, [p1, p2, p3], GameEnv.COLOR_VISION_CONE)

        # Body
        rect = self.get_rect()
        glow_rect = rect.inflate(8, 8)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*GameEnv.COLOR_GUARD_GLOW, 120), glow_surf.get_rect(), border_radius=4)
        screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(screen, GameEnv.COLOR_GUARD, rect, border_radius=4)

class Note:
    def __init__(self, x, y, note_type):
        self.x, self.y = x, y
        self.type = note_type
        self.size = 8

    def get_rect(self):
        return pygame.Rect(self.x - self.size, self.y - self.size, self.size * 2, self.size * 2)

    def draw(self, screen):
        color = GameEnv.NOTE_COLORS[self.type]
        glow_color = (*color, 60)
        pygame.gfxdraw.filled_circle(screen, int(self.x), int(self.y), self.size + 4, glow_color)
        pygame.gfxdraw.aacircle(screen, int(self.x), int(self.y), self.size + 4, glow_color)
        pygame.gfxdraw.filled_circle(screen, int(self.x), int(self.y), self.size, color)
        pygame.gfxdraw.aacircle(screen, int(self.x), int(self.y), self.size, color)

class Particle:
    def __init__(self, x, y, color, rng, speed_mult=1):
        self.x, self.y = x, y
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1, 4) * speed_mult
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.lifespan = rng.integers(15, 30)
        self.max_lifespan = self.lifespan
        self.radius = rng.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.95
        self.vy *= 0.95
        self.lifespan -= 1

    def draw(self, screen):
        alpha = int(255 * (self.lifespan / self.max_lifespan))
        color = (*self.color, alpha)
        temp_surf = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, int(self.radius), int(self.radius), int(self.radius), color)
        screen.blit(temp_surf, (int(self.x - self.radius), int(self.y - self.radius)))

if __name__ == '__main__':
    # --- Manual Play Testing ---
    # This block needs a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Rhythm Jumper - Manual Test")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        movement = 0 # Corresponds to NO-OP
        space_held = 0
        
        # This part of the logic is for manual testing, not for the agent
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        action = [movement, space_held, 0] # Shift is unused in this manual test
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()
    pygame.quit()